import torch
from torch import nn
from transformers import T5Model


class TASTEModel(T5Model):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def wrap_encoder(self, use_checkpoint=False):
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        self.encoder = self.encoder.encoder
        self.encoder.block = nn.ModuleList([mod.module for mod in self.encoder.block])

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def encode(self, input_ids, attention_mask):
        if input_ids.dim() == 3:
            self.encoder.n_passages = input_ids.size(1)
        input_ids = input_ids.view(input_ids.size(0), -1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)

        decoder_input_ids = torch.zeros(
            (input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device
        )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

        hidden = output.last_hidden_state
        reps = hidden[:, 0, :]

        return hidden, reps

    def forward(self, *input):
        return self.encode(*input)


class EncoderWrapper(nn.Module):

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
        self.n_passages = None
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages

        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)

        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        outputs.last_hidden_state = outputs[0].view(
            bsz, self.n_passages * passage_length, -1
        )
        return outputs


class CheckpointWrapper(nn.Module):

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                return tuple(
                    (
                        x
                        if x is not None
                        else torch.tensor(
                            [], dtype=torch.float, device=output[0].device
                        )
                    )
                    for x in output
                )

            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias
            )
            output = tuple(x if x.numel() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    t5stack.block = nn.ModuleList(
        [CheckpointWrapper(mod, use_checkpoint) for mod in t5stack.block]
    )
