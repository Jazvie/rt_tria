import dac
from audiotools import AudioSignal
import time
import argparse
import logging
from pathlib import Path

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
