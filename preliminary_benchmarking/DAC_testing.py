import dac
from audiotools import AudioSignal
import time
import argparse
import logging
from pathlib import Path
import torch

class StreamingWindowBuffer:
    def __init__(self, channels: int, capacity: int, device="cpu", dtype=torch.float32):
        self.buf = torch.zeros(channels, capacity, device=device, dtype=dtype)
        self.capacity = capacity
        self.write_pos = 0          # physical index in circular array
        self.total_written = 0      # absolute sample count
        self.next_start = 0         # absolute start of next inference window

    def push(self, x):
        # x: [C, T]
        n = x.shape[-1]

        if n >= self.capacity:
            x = x[:, -self.capacity:]
            n = x.shape[-1]

        end = self.write_pos + n
        if end <= self.capacity:
            self.buf[:, self.write_pos:end] = x
        else:
            first = self.capacity - self.write_pos
            self.buf[:, self.write_pos:] = x[:, :first]
            self.buf[:, :end % self.capacity] = x[:, first:]

        self.write_pos = end % self.capacity
        self.total_written += n

    def has_enough_for(self, window_size):
        return (self.total_written - self.next_start) >= window_size

    def _read_absolute(self, start, length):
        # start is an absolute stream position
        oldest_available = max(0, self.total_written - self.capacity)
        if start < oldest_available:
            raise ValueError("requested data already overwritten")
        if start + length > self.total_written:
            raise ValueError("requested data not written yet")

        start_idx = start % self.capacity
        end_idx = (start_idx + length) % self.capacity

        if start_idx < end_idx or start_idx + length <= self.capacity:
            return self.buf[:, start_idx:start_idx + length]
        else:
            first = self.capacity - start_idx
            return torch.cat(
                [self.buf[:, start_idx:], self.buf[:, :length - first]],
                dim=-1,
            )

    def read_window(self, window_size):
        return self._read_absolute(self.next_start, window_size)

    def advance(self, hop):
        self.next_start += hop


def demo():
    # Download a model
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)

    model.to('cuda')

    # Load audio signal file
    signal = AudioSignal('input.wav')

    # Encode audio signal as one long file
    # (may run out of GPU memory on long files)
    signal.to(model.device)

    x = model.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, _, _ = model.encode(x)

    # Decode audio signal
    y = model.decode(z)

    # Alternatively, use the `compress` and `decompress` functions
    # to compress long files.

    signal = signal.cpu()
    x = model.compress(signal)

    # Save and load to and from disk
    x.save("compressed.dac")
    x = dac.DACFile.load("compressed.dac")

    # Decompress it back to an AudioSignal
    y = model.decompress(x)

    # Write to file
    y.write('output.wav')

def single_item_streaming_benchmark(input_dir, output_dir, codec_sample_rate):
    #run with batch size 1 and feed one at a time
    #for teach we should compute t_arrival, t_dispatch, t_first_output, t_done
    #with those we can compute queue_wait, time_to_first, service_time, and end_to_end
    pass

def sustained_stream_benchmark(input_dir, output_dir, codec_sample_rate):
    #Test with fix-ed rate arrivals, bursty arrivals
    pass

def batch_sensitivity(input_dir, codec_sample_rate, output_dir):
    #test at barches 1,2,4,4,8. if only works good at high batches then that is probably not good for us
    pass    



def main():
   #don't forget warmup runs 
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_sample_rate", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--save_outputs", type=int, default=0)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)


    if not input_dir.exists() or not input_dir.is_dir():
        raise Exception(f"Input directory does not exist: {input_dir}")
    if args.save_outputs and (not output_dir.exists() or not output_dir.is_dir()):
        raise Exception(f"Output directory does not exist: {output_dir}")

    # Validate codec_sample_rate
    valid_sample_rates = {"44khz", "24khz", "16khz"}
    if args.codec_sample_rate not in valid_sample_rates:
        raise ValueError(
            f"Invalid codec_sample_rate: {args.codec_sample_rate}. "
            f"Supported values are: {', '.join(valid_sample_rates)}")



    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting...")




if __name__=="__main__":
    main()
