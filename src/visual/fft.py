import numpy as np
import matplotlib.pyplot as plt

def graph_fft(x_values, y_values, title: str="", save_fig: bool=False, show_fig: bool=True):
  """
  Displays/saves the graph for FFT transform in the frequency domain.
  By default, only display fig, does not save.
  Note that the unit for amplitude is in DB (therefore it is logarithmic).
  """
  
  db = 20 * np.log10(np.abs(y_values) + 1e-12)  # Add a small value to avoid log(0)

  # Plot the frequency domain graph (amplitude spectrum in dB)
  plt.figure(figsize=(10, 6))
  plt.plot(x_values, db)
  
  # Limit x-axis and y-axis to positive values only
  plt.xlim(left=0)  # Set x-axis to start from 0 (positive frequencies only)
  plt.ylim(bottom=0)  # Set y-axis to start from 0 (positive amplitude only)
  
  plt.title(f'Frequency Domain Representation of {title} Signal (Amplitude in dB)')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Amplitude (dB)')
  plt.grid(True)
  
  if show_fig:
    plt.show()
  
  if save_fig:
    """
    savefig(fname, *, transparent=None, dpi='figure', format=None,
          metadata=None, bbox_inches=None, pad_inches=0.1,
          facecolor='auto', edgecolor='auto', backend=None,
          **kwargs
    )
    """
    plt.savefig()
  
  return

def fft(data: list, sampling_rate: int=1e9) -> list[list, list]:
  """
  Uses numpy to compute the FFT of a signal
  Returns the fft_signal and its respective freqs as a list
  
  Params
  ----------
  data: list, required
    A list of variables
  sampling_rate: int, required
    The sample rate (in HZ) of the signal
  
  Returns
  ----------
  [ fft_signal: NDArray[floating[Any]], freqs: NDArray[floating[Any]] ]
  """
  
  
  # Step 1: Define your signal and sampling rate
  signal = np.array(data)
  N = len(signal)
  
  # Step 2: apply the FFT (fast fourier transform) to transform the data into the frequency domain
  fft_signal = np.fft.fft(signal)
  fft_signal = np.fft.fftshift(fft_signal)
  
  # Step 3: Calculate the frequency axis
  freqs = np.fft.fftfreq(N, 1/sampling_rate)
  freqs = np.fft.fftshift(freqs)  # Shift frequencies to match FFT shift
  
  return [fft_signal, freqs]

def main():
  from src.data.file_handler import read_file
  from src.data.processing import parse_str

  array_data_1 = parse_str(read_file("./dataset/data64QAM.txt"))
  array_data_2 = parse_str(read_file("./dataset/OSC_sync_473.txt"))
  
  fft_signal_1, freqs_1 = fft(array_data_1)
  fft_signal_2, freqs_2 = fft(array_data_2)
  graph_fft(freqs_1, fft_signal_1, show_fig=False)
  graph_fft(freqs_2, fft_signal_2, show_fig=False)
  plt.show()


if __name__ == "__main__":
  main()