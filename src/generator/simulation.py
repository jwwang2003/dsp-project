

#%%
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from scipy.signal import upfirdn, convolve
from scipy.fftpack import fft, ifft, fftshift

################################################################################
###########                  Helper functions                 ##################
################################################################################
def read_constellation_file(order: int, constellation: str) -> np.ndarray:
    """
    Reads the constellation points from a file.

    Args:
        QAMorder (int): Modulation order (e.g., 64 for 64-QAM).
        constellation (str): Constellation type (e.g., 'QAM').

    Returns:
        ndarray: Array of complex constellation points.
    """
    namepre = 'CCSDS'
    namepost = '.txt'
    filename = f"{namepre}{order}{constellation}{namepost}"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Read the data from the file
        data = np.loadtxt(path.join(current_dir, "sample", filename))

        # Create complex constellation points
        cons = data[:, 0] + 1j * data[:, 1]
        return cons
    except:
        print("Something went wrong while trying to read constellation file!")


def QAM_mod(data: np.ndarray, M: int) -> np.ndarray:
    """
    QAM modulation of input data.

    Args:
        data (ndarray): Array of integer symbols (from 0 to M-1).
        M (int): Modulation order (e.g., 16 for 16-QAM).

    Returns:
        ndarray: Modulated complex symbols.
    """
    k = int(np.log2(M))
    sqrt_M = int(np.sqrt(M))
    I = 2 * (data % sqrt_M) - sqrt_M + 1
    Q = 2 * (data // sqrt_M) - sqrt_M + 1
    constellation_points = (I + 1j * Q) / np.sqrt(2 / 3 * (M - 1))
    return constellation_points

def QAM_mod_from_cons(decdata: np.ndarray, cons: np.ndarray) -> np.ndarray:
    """
    Get the QAM modulation of input data by reading a set of
    constellation points from an external file.

    Args:
        decdatat (ndarray): Array of integer symbols (from 0 to M-1).
        cons (ndarray): The constellation mapping for the signal type.

    Returns:
        ndarray: Modulated complex symbols.
    """
    # Ensure that the symbol index does not exceed that number of constellation points
    if np.any(decdata < 0) or np.any(decdata >= len(cons)):
        print(len(cons))
        raise ValueError("Data symbols out of range.")
    qamdata = cons[decdata]
    return qamdata

def upsample(x, up):
    """
    Upsamples the input signal by inserting zeros between samples.
    Another implementation of upsample because the built-in method causes the
    output to be a few bits short.

    Args:
        x (ndarray): Input signal.
        up (int): Upsampling factor.

    Returns:
        ndarray: Upsampled signal.
    """
    N = len(x)
    upsampled = np.zeros(N * up, dtype=x.dtype)
    upsampled[::up] = x
    return upsampled

################################################################################
###########                   Main Functions                  ##################
################################################################################
def generate_seq(
        order: int,
        constellation: str,
        num_symbols: int,
        embedded: bool=False,
        file_embedded: str="",
        up_sample: int=4,
        seed: int=2,

        preamble_length: int=32
    ):
    """
    Generates a sequence of random modulated signals (non-attenuated)
    """
    np.random.seed(seed)

    if np.log2(order) % 1 != 0:
        raise ValueError("QAM order must be a power of 2.")

    # (1) First read from the constellation file to ensure QAM order is valid
    cons = read_constellation_file(order, constellation)

    file_symbols: np.ndarray = []
    # (2) Then process the embedded file, it required
    if embedded:
        # Step 1: Read file and convert to binary
        try:
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            current_dir = os.getcwd()
            file_path = path.join(current_dir, file_embedded)
            print(file_path)
            with open(file_path, "rb") as f:
                file_data = f.read()
            file_binary = np.unpackbits(np.frombuffer(file_data, dtype=np.uint8))
        except:
            raise("Embedded file not found!")
        
        # Step 2: Generate QAM representation of the file binary
        bits_per_symbol = int(np.log2(order))
        # Ensure that the binary data size is divisible by bits_per_symbol
        padding_length = (bits_per_symbol - len(file_binary) % bits_per_symbol) % bits_per_symbol
        padded_binary = np.pad(file_binary, (0, padding_length), constant_values=0)

        # Debugging print statements
        # print(f"Original binary length: {len(file_binary)}")
        # print(f"Padded binary length: {len(padded_binary)}")
        # print(f"Bits per symbol: {bits_per_symbol}")

        # Reshape and convert to symbols
        file_symbols = padded_binary.reshape(-1, bits_per_symbol).dot(1 << np.arange(bits_per_symbol)[::-1])
    
    ############################################################################
    ###########        (3) Preparing the transmission seq.        ##############
    ############################################################################
    # File QAM data
    file_qamdata = None
    if embedded:
        file_qamdata = QAM_mod_from_cons(file_symbols, cons)

    # Generate a preamble sequence and a known EOT (end-of-transmission) sequence
    # Generate synchronization preamble
    # preamble_length = 32  # Adjust as needed
    preamble_symbols = np.random.randint(0, order, preamble_length)
    preamble_qamdata = QAM_mod_from_cons(preamble_symbols, cons)
    preamble_upsampled = upsample(preamble_qamdata, up_sample)
    # Generate a known sequence of EOT
    eot_symbols = np.array([order - 1] * 10)  # A simple implementation 10 consecutive symbols at the maximum value
    eot_qamdata = QAM_mod_from_cons(eot_symbols, cons)
    eot_upsampled = upsample(eot_qamdata, up_sample)

    # Generate the rest of the random simulated QAM siganl sequence
    rand_data = np.random.randint(0, order, num_symbols)
    rand_qamdata = QAM_mod_from_cons(rand_data, cons)

    # (5) Pick a place to insert data to transmit
    # Parameters for the normal distribution
    mu = num_symbols/2          # Mean of the distribution
    sigma = 10                  # Standard deviation

    # Generate a random number from the normal distribution
    random_number = np.random.normal(mu, sigma)

    # Ensure the number is within a specific range
    rand_insertion_index = -1
    while True and embedded:
        min_val = 0
        max_val = num_symbols
        rand_insertion_index = int(np.clip(random_number, min_val, max_val))
        # Ensure the selected index + length of file QAM and sync seq does
        # not exceed the end lenght of the randomly generated data
        if rand_insertion_index + len(file_qamdata) < num_symbols:
            print(f"Selected encoded file insertion index: {rand_insertion_index}")
            break

    # (5) Construct the final data sequence and upsample the signal!
    if embedded:
        valid_datalength = len(preamble_qamdata) + len(file_qamdata) + len(eot_qamdata)
        # Embed the valid data sequence into the random generated signal
        rand_qamdata = np.concatenate(
            [
                rand_qamdata[:rand_insertion_index],
                preamble_qamdata,
                file_qamdata,
                eot_qamdata,
                rand_qamdata[valid_datalength + rand_insertion_index:]
            ]
        )
    # Ensure that the newly generated signal sequence is the same length as the
    # required num_symbols specificed by the function's parameters
    assert len(rand_qamdata) == num_symbols

    # data_afterUpsample = upfirdn([1], qamdata, up=upsampleno)
    data_afterUpsample = upsample(rand_qamdata, up_sample)

    # Generate shaping filter
    # What is the purpose of the shapping filte?
    upsamplesymbol = num_symbols * up_sample  # Number of symbols after upsampling
    t_sequence_norm = np.arange(upsamplesymbol) - upsamplesymbol / 2

    alpha = 0.205
    subcar = 1 / 2
    startFreq = 1 / 100

    # Calculate gtr
    pi_t_alpha = np.pi * t_sequence_norm * (1 + alpha) * 0.25
    pi_t_alpha_neg = np.pi * t_sequence_norm * (1 - alpha) * 0.25
    denominator = 4 * alpha * t_sequence_norm * 0.25

    # Handle division by zero
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    gtr1 = np.cos(pi_t_alpha) + np.sin(pi_t_alpha_neg) / denominator
    gtr2 = gtr1 / (1 - (4 * alpha * t_sequence_norm * 0.25) ** 2)
    gtr = gtr2 * 4 * alpha / np.pi
    gtr[int(upsamplesymbol / 2)] = (1 + alpha * (4 / np.pi - 1))

    BW = 1 + alpha
    subcar1 = BW * subcar + startFreq

    filter_I_path = gtr * np.cos(2 * np.pi * subcar1 * t_sequence_norm * 0.25)
    filter_Q_path = gtr * np.sin(2 * np.pi * subcar1 * t_sequence_norm * 0.25)

    filter_I_path = filter_I_path.reshape(-1, 1)
    filter_Q_path = filter_Q_path.reshape(-1, 1)

    # Pulse shaping
    taps = 35  # Filter length (should be odd number)

    gtI1 = filter_I_path.flatten()
    gtQ1 = filter_Q_path.flatten()
    Idata1 = np.real(data_afterUpsample).flatten()
    Qdata1 = np.imag(data_afterUpsample).flatten()

    center_index = int(numofsymbols * up_sample / 2 - (taps - 1) / 2)
    filterI = gtI1[center_index: center_index + taps]
    filterQ = gtQ1[center_index: center_index + taps]
    
    # Convolve data with filters
    DataCapI = convolve(Idata1, filterI, mode='same')
    DataCapQ = convolve(Qdata1, filterQ, mode='same')
    
    output_data = DataCapI - DataCapQ

    return output_data


def signal_attenuation():
    pass

################################################################################
###########                   Visualization                   ##################
################################################################################
# def graph


################################################################################
###########                    DEMO DRIVER                    ##################
################################################################################

def generate_random_seq_with_file_embedded():
    QAMorder = 64
    constellation = 'QAM'
    numofsymbols = 51200

    output_data = generate_seq(order=64, constellation="QAM", embedded=True, file_embedded="test_data.txt", num_symbols=numofsymbols)
    dataout = output_data / np.sqrt(np.mean(np.abs(output_data) ** 2))
    dataout = dataout.reshape(-1, 1)

    # Plot Frequency Response
    dataout_flat = dataout.flatten()
    awg_SampleRate = 1e7  # Adjust as needed
    xfre = ((np.arange(len(dataout_flat)) - len(dataout_flat) / 2) * awg_SampleRate / 1e6) / len(dataout_flat)
    # DataCap_f = fftshift(fft(dataout_flat))
    DataCap_f = np.fft.fft(dataout_flat)
    fft_signal = np.fft.fftshift(DataCap_f)

    print(len(xfre), len(DataCap_f))

    plt.figure(figsize=(10, 6))
    # Limit x-axis and y-axis to positive values only
    plt.xlim(left=0, right=5)  # Set x-axis to start from 0 (positive frequencies only)
    plt.ylim(bottom=0, top=80)  # Set y-axis to start from 0 (positive amplitude only)
    print(plt.xlim())
    print(plt.ylim())
    plt.plot(xfre, 20 * np.log10(np.abs(fft_signal) + 1e-12), color="blue")
    plt.title(f'Overlap of Two FFT Signals and Region of Difference\n')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def generate_random_seq():
    QAMorder = 64
    constellation = 'QAM'
    numofsymbols = 51200

    output_data = generate_seq(order=64, constellation="QAM", num_symbols=numofsymbols)
    dataout = output_data / np.sqrt(np.mean(np.abs(output_data) ** 2))
    dataout = dataout.reshape(-1, 1)

    # Plot Frequency Response
    dataout_flat = dataout.flatten()
    awg_SampleRate = 1e7  # Adjust as needed
    xfre = ((np.arange(len(dataout_flat)) - len(dataout_flat) / 2) * awg_SampleRate / 1e6) / len(dataout_flat)
    # DataCap_f = fftshift(fft(dataout_flat))
    DataCap_f = np.fft.fft(dataout_flat)
    fft_signal = np.fft.fftshift(DataCap_f)

    print(len(xfre), len(DataCap_f))

    plt.figure(figsize=(10, 6))
    # Limit x-axis and y-axis to positive values only
    plt.xlim(left=0, right=5)  # Set x-axis to start from 0 (positive frequencies only)
    plt.ylim(bottom=0, top=80)  # Set y-axis to start from 0 (positive amplitude only)
    print(plt.xlim())
    print(plt.ylim())
    plt.plot(xfre, 20 * np.log10(np.abs(fft_signal) + 1e-12), color="blue")
    plt.title(f'Overlap of Two FFT Signals and Region of Difference\n')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    generate_random_seq()
    pass

# # Modulation order
# M = 64  # For 16-QAM

# # Number of symbols to generate
# num_symbols = 1000

# # Generate random data symbols between 0 and M-1
# np.random.seed(0)  # For reproducibility
# data_symbols = np.random.randint(0, M, num_symbols)

# modulated_symbols = QAM_mod(data_symbols, M)
# print(modulated_symbols)

# # Plot the constellation diagram
# plt.figure(figsize=(8, 8))
# plt.plot(np.real(modulated_symbols), np.imag(modulated_symbols), 'bo', markersize=2)
# plt.title(f'{M}-QAM Constellation Diagram')
# plt.xlabel('In-phase Component')
# plt.ylabel('Quadrature Component')
# plt.grid(True)
# plt.axis('square')
# plt.show()
# %%
