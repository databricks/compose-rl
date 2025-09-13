# Copyright 2025 Databricks Mosaic Research
# SPDX-License-Identifier: Apache-2.0

"""VLLMMinievalCallback for evaluating models on GSM8K during training."""
import concurrent.futures
import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any, Callable

import ray
import torch
from composer.core import State, Time, TimeUnit
from composer.devices import Device
from composer.loggers import Logger, MLFlowLogger, WandBLogger
from composer.utils import dist, ensure_tuple
from llmfoundry.interfaces import CallbackWithConfig
from minieval import common as mcommon
from minieval import loggers as mloggers
from minieval import registry as mregistry
from minieval import state as mstate
from minieval.models.generative_model import (
    GenerativeModelInput,
    GenerativeModelOutput,
)
from minieval.models.model import Model
from minieval.utils import builders, config_utils
from omegaconf import DictConfig

from compose_rl.algorithms.online.callback import OnPolicyCallback

log = logging.getLogger(__name__)


class VLLMEngineMinievalCallback(CallbackWithConfig):

    def __init__(
        self,
        evals: dict[str, str],
        train_config: DictConfig,
        eval_overrides: dict[str, str] | None = None,
    ):
        """Initialize the VLLMMinievalCallback.

        Args:
            evals: The evals to run.
            train_config: The configuration for the callback.
                Should include the vllm_engines used for inference.
            eval_overrides: The overrides for the evals. Default is None.
        """
        self.evals = evals
        self.eval_overrides = eval_overrides
        self.eval_interval = self._get_eval_interval(train_config)
        self.max_duration = train_config.get('max_duration')

        python_log_level = train_config.get('python_log_level', None)
        if python_log_level is not None:
            log.setLevel(python_log_level.upper())

        self.mlflow_logger = None

        # Create a ThreadPoolExecutor with a single worker
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future_handle = None

    def _get_eval_interval(self, train_config: DictConfig) -> bool:
        eval_interval = train_config.get('eval_interval', 1)
        time_eval_interval: Time = Time.from_input(
            eval_interval,
            TimeUnit.ITERATION,
        )
        if time_eval_interval.unit != TimeUnit.ITERATION:
            # ComposeRL's OnlineRL framework updates vLLM weights on an iteration
            # time basis; it only makes sense to evaluate on an iteration basis.
            raise ValueError('eval_interval must be in iterations')
        if time_eval_interval.value <= 0:
            raise ValueError('eval_interval must be positive')
        return time_eval_interval.value

    def init(self, state: State, logger: Logger):
        """Get / set mlflow_logger at `init`."""
        del state  # unused

        for destination in ensure_tuple(logger.destinations):
            if isinstance(destination, WandBLogger):
                log.warning(
                    'WandBLogger not supported for Minieval. ' + 'Please use MLFlowLogger instead.',
                )
            if isinstance(destination, MLFlowLogger):
                self.mlflow_logger = destination

    def fit_start(self, state: State, logger: Logger) -> None:
        """Define the MiniEval Model class that integrates with vLLM engines.

        Args:
            state: The composer state object
            logger: The composer logger
        """
        del logger  # unused

        assert hasattr(state, 'vllm_engines') or hasattr(state, 'vllm_client'), 'vLLM engines or client not found in state'
        from orl_servers.client import ArealOpenAI
        from orl_servers.async_utils import run_async_sync

        class OnlineVLLMModel(Model):

            def __init__(self, config: mcommon.BaseConfig):
                super().__init__(config)
                self.vllm_engines = getattr(state, 'vllm_engines', None)  # type: ignore
                self.vllm_client: ArealOpenAI | None = getattr(state, 'vllm_client', None)  # type: ignore

            def __call__(
                self,
                prompts: Iterable[GenerativeModelInput],
                state: mstate.State,
                logger: mloggers.Logger,
            ) -> Iterable[GenerativeModelOutput]:
                all_inputs = list(prompts)
                gen_params = all_inputs[0].generation_params

                if self.vllm_client is not None:
                    all_results = []
                    async def _run_batch():
                        tasks = []
                        for input in all_inputs:
                            task = asyncio.create_task(self.vllm_client.chat.completions.create(
                                messages=input.messages,
                                **gen_params,
                            ))
                            tasks.append(task)
                        results = await asyncio.gather(*tasks)
                        return results
                    all_results = run_async_sync(_run_batch())
                    for result in all_results:
                        all_results.append(GenerativeModelOutput(generation=result.choices[0].message.content))

                assert self.vllm_engines is not None, 'vLLM engines not found in state'
                n = len(self.vllm_engines)
                _size = (len(all_inputs) + n - 1) // n
                cur_inputs = [all_inputs[i * _size:(i + 1) * _size] for i in range(n)]
                assert len(cur_inputs) == len(self.vllm_engines)

                futs = []
                for inputs, vllm_engine in zip(cur_inputs, self.vllm_engines):
                    messages = [input.messages for input in inputs]
                    fut = vllm_engine.chat.remote(
                        messages=messages,
                        sampling_params=gen_params,
                    )
                    futs.append(fut)

                all_results = []
                for result in ray.get(futs):
                    for res in result:
                        all_results.extend([
                            GenerativeModelOutput(
                                generation=res.outputs[0].text,
                            ),
                        ])

                assert len(all_inputs) == len(all_results)
                return all_results

        mregistry.models.register(
            name='online-vllm',
            func=(OnlineVLLMModel, mcommon.BaseConfig),
        )

    def epoch_start(self, state: State, logger: Logger) -> None:
        """Run evaluation at specified intervals before the batch is trained thru the mode.

        Args:
            state: The composer state object
            logger: The composer logger
        """
        del logger  # unused

        # Check if it's time to run evaluation
        cur_itr = state.timestamp.iteration.value
        if cur_itr % self.eval_interval == 0:
            log.info(f'Running evaluation at iteration {cur_itr}.')

            expr, expr_name, step = None, None, None
            if self.mlflow_logger is not None:
                expr = self.mlflow_logger.experiment_name
                expr_name = self.mlflow_logger.run_name
                step = state.timestamp.batch.value

            self._launch_fn_r0_detached(
                self._run_evaluation,
                expr,
                expr_name,
                step,
            )

            # Synchronize all processes
            log.info('Running dist barrier after launch.')
            dist.barrier()

    def iteration_end(self, state: State, logger: Logger) -> None:
        """Get future handle result and sync GPUs after epoch start async runs evals.

        Args:
            state: The composer state object
            logger: The composer logger
        """
        del logger  # unused

        device = state.device

        log.debug('Waiting for completion at iteration_end.')
        self._sync_gpu_on_future_done(device)
        log.debug('Future completed at iteration_end.')

        dist.barrier()
        log.debug('Barrier completed at iteration_end.')

    def fit_end(self, state: State, logger: Logger) -> None:
        """Run evaluation at end of training.

        Args:
            state: The composer state object
            logger: The composer logger
        """
        del logger  # unused

        device = state.device

        log.info('Updating the inference weights in vLLM at fit_end.')

        # update the inference weights in vLLM
        ppo_callback = None
        for callback in state.callbacks:
            if isinstance(callback, OnPolicyCallback):
                ppo_callback = callback
                break
        batch = device.batch_to_device(ppo_callback._get_next_iter_prompts())
        if ppo_callback.vllm_engines is not None:
            ppo_callback._update_inference_model(batch)

        log.info('Running evaluation at fit_end.')

        # Run the evaluation (this will be run from rank 0)
        expr, expr_name = None, None
        if self.mlflow_logger is not None:
            expr = self.mlflow_logger.experiment_name
            expr_name = self.mlflow_logger.run_name
        step = state.timestamp.batch.value

        self._launch_fn_r0_detached(
            self._run_evaluation,
            expr,
            expr_name,
            step,
        )
        # Synchronize all processes
        log.info('Running dist barrier after launch at fit end.')
        dist.barrier()

        log.debug('Waiting for completion at fit_end.')
        self._sync_gpu_on_future_done(device)
        log.debug('Future completed at fit_end.')

        dist.barrier()
        log.debug('Barrier completed at fit_end.')

        # Shutdown the executor
        self.executor.shutdown()
        self.executor = None
        log.debug('Executor shutdown at fit_end.')

    def _run_evaluation(self, expr: str, expr_name: str, step: int):
        """Run the evaluation with the specified model using the MiniEval runner."""
        config = {
            'model': {
                'name': 'online-vllm',
            },
            'evals': self.evals,
        }

        if self.eval_overrides is not None:
            config['eval_overrides'] = self.eval_overrides

        if self.mlflow_logger is not None:
            config['loggers'] = {}
            config['loggers']['mlflow'] = {  # type: ignore[reportGeneralTypeIssues]
                'experiment': expr,
                'name': expr_name,
                'step': step,
                'keep_run_alive': True,
            }
            config[  # type: ignore[reportGeneralTypeIssues]
                'save_folder'] = f'mlflow://{expr}/{expr_name}/artifacts/minieval/step_{step:04d}'

        runner = builders.build_runner_from_config(
            config=config_utils.EvalRunnerConfig(
                **config,  # type: ignore[reportGeneralTypeIssues]
            ),
        )
        runner.run()

    def _launch_fn_r0_detached(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future:  # type: ignore[reportGeneralTypeIssues]
        """Launch a function in a detached thread.

        Args:
            fn: The function to run
            *args: The arguments to pass to the function
            **kwargs: The keyword arguments to pass to the function
        """
        # Run the evaluation (this will be run from rank 0)
        # launch asynchronous process with will run in the background
        # and synchronize with the main process at the end of iteration
        if dist.get_global_rank() == 0:
            # Submit the function to the executor
            self.future_handle = self.executor.submit(fn, *args, **kwargs)

    def _sync_gpu_on_future_done(self, device: Device) -> None:
        """Sync GPU on future done.

        This function is called when the future is done to synchronize the GPUs.
        """
        done = False
        while not done:
            # Check if the future is done
            # dist.barrier() cannot naively be used here since it
            # will timeout for evals that take longer than the timeout
            done_check = device.tensor_to_device(
                torch.tensor([1], dtype=torch.uint8),
            )
            if self.future_handle is not None and not self.future_handle.done():
                done_check.fill_(0)

            # Broadcast the done_check tensor from rank 0 to all processes
            dist.broadcast(done_check, src=0)
            done = bool(done_check.item())
            if done:
                if self.future_handle is not None:
                    # Only True on rank 0
                    try:
                        # Wait for the future to complete and get the result (if any)
                        self.future_handle.result()
                    except Exception as e:
                        log.error(
                            f'Main: Eval process future failed with error: {e}',
                        )
                self.future_handle = None
            else:
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)
