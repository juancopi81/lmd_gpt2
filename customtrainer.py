from transformers import Trainer
import wandb
import note_seq

from utils import token_sequence_to_note_sequence

# first create a custom trainer to log prediction distribution
SAMPLE_RATE = 44100


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # call super class method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # log the prediction distribution using `wandb.Histogram` method.
        if wandb.run is not None:
            input_ids = self.tokenizer.encode(
                "PIECE_START STYLE=JSFAKES GENRE=JSFAKES TRACK_START",
                return_tensors="pt",
            ).cuda()
            # Generate more tokens.
            voice1_generated_ids = self.model.generate(
                input_ids,
                max_length=2048,
                do_sample=True,
                temperature=0.75,
                eos_token_id=self.tokenizer.encode("TRACK_END")[0],
            )
            voice2_generated_ids = self.model.generate(
                voice1_generated_ids,
                max_length=2048,
                do_sample=True,
                temperature=0.75,
                eos_token_id=self.tokenizer.encode("TRACK_END")[0],
            )
            voice3_generated_ids = self.model.generate(
                voice2_generated_ids,
                max_length=2048,
                do_sample=True,
                temperature=0.75,
                eos_token_id=self.tokenizer.encode("TRACK_END")[0],
            )
            voice4_generated_ids = self.model.generate(
                voice3_generated_ids,
                max_length=2048,
                do_sample=True,
                temperature=0.75,
                eos_token_id=self.tokenizer.encode("TRACK_END")[0],
            )
            token_sequence = self.tokenizer.decode(voice4_generated_ids[0])
            note_sequence = token_sequence_to_note_sequence(token_sequence)
            synth = note_seq.fluidsynth
            array_of_floats = synth(note_sequence, sample_rate=SAMPLE_RATE)
            int16_data = note_seq.audio_io.float_samples_to_int16(array_of_floats)
            wandb.log({"Generated_audio": wandb.Audio(int16_data, SAMPLE_RATE)})

        return eval_output
