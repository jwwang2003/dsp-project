#%%
import numpy as np
import matplotlib.pyplot as plt

def graph_fft_overlap(
      freqs_1, fft_signal_1, freqs_2, fft_signal_2, 
      title: str = "", signal_1_name: str="Signal 1", signal_2_name: str="Signal 2", save_fig: bool = False, show_fig: bool = True
    ):
    """
    Displays/saves the graph for overlapping FFT transforms in the frequency domain.
    Highlights the differences between two FFT signals.
    """
    # Convert amplitude to dB scale
    db_1 = 20 * np.log10(np.abs(fft_signal_1) + 1e-12)  # Avoid log(0)
    db_2 = 20 * np.log10(np.abs(fft_signal_2) + 1e-12)  # Avoid log(0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot both FFT signals
    plt.plot(freqs_1, db_1, label=f"{signal_1_name} (dB)", color="gray", alpha=0.75)
    plt.plot(freqs_2, db_2, label=f"{signal_2_name} (dB)", color="blue")

    # Limit x-axis and y-axis to positive values only
    plt.xlim(left=0)  # Set x-axis to start from 0 (positive frequencies only)
    plt.ylim(bottom=0)  # Set y-axis to start from 0 (positive amplitude only)

    # Highlight the region of difference
    # plt.fill_between(freqs_1, db_1, db_2, where=(db_1 > db_2), interpolate=True, color='gray', alpha=0.3, label='Difference Region (Signal 1 > Signal 2)')
    # plt.fill_between(freqs_1, db_1, db_2, where=(db_2 > db_1), interpolate=True, color='yellow', alpha=0.3, label='Difference Region (Signal 2 > Signal 1)')

    # Add labels, legend, and title
    plt.title(f'{title}\n')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend(loc='best')
    plt.grid(True)

    # Show and/or save the figure
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(f"{title.replace(' ', '_')}_FFT_Comparison.png", bbox_inches="tight")


def graph_log(x_values, y_values, title: str="", save_fig: bool=False, show_fig: bool=True):
  """
  Displays/saves the graph for FFT transform in the frequency domain.
  By default, only display fig, does not save.
  Note that the unit for amplitude is in DB (therefore it is logarithmic).
  """
  
  db = 20 * np.log10(np.abs(y_values) + 1e-12)  # Add a small value to avoid log(0)
  plt.figure(figsize=(10, 6))
  # Plot the frequency domain graph (amplitude spectrum in dB)
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
  graph_log(freqs_1, fft_signal_1, show_fig=False)
  graph_log(freqs_2, fft_signal_2, show_fig=False)
  plt.show()


if __name__ == "__main__":
  main()
#%%