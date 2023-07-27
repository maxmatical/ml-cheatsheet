from transformers import Trainer, PreTrainedModel
from rlhf.data.data import RewardDataCollatorWithPadding
from torch import nn
from rlhf.optimizer.lion import DecoupledLionW
from transformers.trainer_pt_utils import get_parameter_names 
import torch
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


class RewardTrainer(Trainer):
    def __init__(
        self, 
        model: PreTrainedModel,
        args,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        max_length: int = 512,
        use_lion: bool = False
    ):
        # data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.use_lion = use_lion

    def compute_loss(self, model, inputs):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"]
        )
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        return loss


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        # lion or adam
        if not self.use_lion:
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        else:
            optimizer_cls = DecoupledLionW
            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
            }
        

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        # print(f"Using optimizer {self.optimizer}")

        return self.optimizer
    