__author__ = "JUN WEI WANG"
__email__ = "wjw_03@outlook.com"

#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift
import os
from os import path

# from scipy.signal import upfirdn  # Re-implemented

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
        print(path.join(current_dir, "sample", filename))
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
        raise ValueError("Data symbols out of range.")
    qamdata = cons[decdata]
    return qamdata

def upsample(x: np.ndarray, up: int):
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

def downsample(x: np.ndarray, down: int):
    """
    Downsamples the input signal by selecting every 'down'-th sample.
    Custom implementation of downsampling to ensure precise control over the output length,
    especially when built-in methods may cause the output to be a few samples off.

    Args:
        x (ndarray): Input signal.
        down (int): Downsampling factor.

    Returns:
        ndarray: Downsampled signal.
    """
    if down <= 0:
        raise ValueError("Downsampling factor must be a positive integer.")
    downsampled = x[::down]
    return downsampled

def get_filter_paths(
    num_symbols:    int,
    up_sample:      int     = 4,
    alpha:          float   = 0.205,
    subcar:         float   = 1 / 2,
    startFreq:      float   = 1 / 100,
) -> np.ndarray:
    """
    Generate the In-phase (I) and Quadrature (Q) shaping filter paths for pulse shaping.

    This function computes the I and Q filter paths used for pulse shaping in digital communication
    systems, specifically for Single Carrier Amplitude and Phase modulation (CAP) or similar schemes.
    The filters are based on a raised cosine function with a specified roll-off factor and are
    modulated with cosine and sine carriers at a specified subcarrier frequency.

    Parameters:
        num_symbols (int): The number of symbols in the signal sequence.
        up_sample (int, optional): The upsampling factor applied to the signal. Default is 4.
        alpha (float, optional): The roll-off factor of the raised cosine filter (0 < alpha <= 1). Default is 0.205.
        subcar (float, optional): The normalized subcarrier frequency (relative to the symbol rate). Default is 0.5.
        startFreq (float, optional): The starting frequency offset (relative to the symbol rate). Default is 0.01.

    Returns:
        tuple:
            - filter_Q_path (np.ndarray): The Quadrature (Q) filter path coefficients as a column vector.
            - filter_I_path (np.ndarray): The In-phase (I) filter path coefficients as a column vector.

    Notes:
        - The shaping filters are designed to control the bandwidth of the transmitted signal and reduce inter-symbol interference (ISI).
        - The filters are based on a raised cosine function modulated by cosine and sine functions to shift the spectrum.
        - The filters are computed over a time sequence normalized to the symbol period, taking into account the upsampling factor.

    Example:
        >>> filter_Q, filter_I = get_filter_paths(num_symbols=1000, up_sample=4)
        >>> print(filter_I.shape)
        (4000, 1)
        >>> print(filter_Q.shape)
        (4000, 1)

    """
    # Generate shaping filter
    upsamplesymbol = num_symbols * up_sample  # Number of symbols after upsampling
    t_sequence_norm = (
        np.linspace(1, upsamplesymbol, upsamplesymbol) - 1 - upsamplesymbol / 2
    )

    # Calculate gtr (time-domain impulse response of the shaping filter)
    pi_t_alpha = np.pi * t_sequence_norm * (1 + alpha) * 0.25
    pi_t_alpha_neg = np.pi * t_sequence_norm * (1 - alpha) * 0.25
    denominator = 4 * alpha * t_sequence_norm * 0.25

    # Handle division by zero for the denominator
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)

    # Compute the raised cosine filter components
    gtr1 = np.cos(pi_t_alpha) + np.sin(pi_t_alpha_neg) / denominator
    gtr2 = gtr1 / (1 - (4 * alpha * t_sequence_norm * 0.25) ** 2)
    gtr = gtr2 * 4 * alpha / np.pi

    # Correct the center value to avoid NaN or Inf
    center_index = int(upsamplesymbol / 2)
    gtr[center_index] = (1 + alpha * (4 / np.pi - 1))

    # Bandwidth and subcarrier frequency adjustments
    BW = 1 + alpha
    subcar1 = BW * subcar + startFreq

    # Generate the I and Q filter paths by modulating with cosine and sine carriers
    filter_I_path = gtr * np.cos(2 * np.pi * subcar1 * t_sequence_norm * 0.25)
    filter_Q_path = gtr * np.sin(2 * np.pi * subcar1 * t_sequence_norm * 0.25)

    # Reshape the filters to column vectors
    filter_I_path = filter_I_path.reshape(-1, 1)
    filter_Q_path = filter_Q_path.reshape(-1, 1)

    return filter_Q_path, filter_I_path

################################################################################
###########            Signal Attenuation Simulation          ##################
################################################################################

# Helper function for signal attenuation
def awgn(signal: np.ndarray, snr_dB: float):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal to achieve a specified SNR.

    This function simulates the effect of AWGN by adding complex Gaussian noise to the input signal,
    resulting in a noisy signal with the desired signal-to-noise ratio (SNR) in decibels (dB).

    Parameters:
        signal (np.ndarray): The input signal to which noise will be added. Can be real or complex.
        snr_dB (float): The desired signal-to-noise ratio in decibels (dB).

    Returns:
        np.ndarray: The noisy signal after adding AWGN, with the same shape as the input signal.

    Notes:
        - The function calculates the signal power and determines the noise power required to achieve the specified SNR.
        - Noise is generated as complex Gaussian noise, ensuring proper noise modeling for both real and complex signals.
        - The noise added has zero mean and is scaled to achieve the desired SNR relative to the input signal.

    Example:
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        >>> noisy_signal = awgn(signal, snr_dB=20)
        >>> print(noisy_signal)
        [1.045 1.998 2.953 4.005 5.023]

    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_dB / 10)
    # Calculate the average power of the signal
    power_signal = np.mean(np.abs(signal) ** 2)
    # Calculate the required noise power for the specified SNR
    power_noise = power_signal / snr_linear
    # Generate complex Gaussian noise
    noise = np.sqrt(power_noise / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    # Add noise to the original signal
    return signal + noise

def signal_attenuation(
        data:       np.ndarray,
        SNR:        int = 20,         # dB
        Fs:         int = 100,
        Factor:     int = 40
    ) -> np.ndarray:
    """
    Simulate signal attenuation and add AWGN noise to the input signal.

    This function models the effect of a frequency-dependent channel attenuation and adds Additive White Gaussian Noise (AWGN)
    to the input signal. The attenuation is applied in the frequency domain using an exponential decay function, and the
    signal is then transformed back to the time domain. Finally, AWGN is added to achieve the specified Signal-to-Noise Ratio (SNR).

    Parameters:
        data (np.ndarray): The input signal data to be attenuated and noised. Should be a one-dimensional array.
        SNR (int, optional): The desired signal-to-noise ratio in decibels (dB). Default is 20 dB.
        Fs (int, optional): The sampling frequency of the signal in Hz. Used to calculate the frequency vector. Default is 100 Hz.
        Factor (int, optional): The attenuation factor controlling the rate of exponential decay in the channel model. Default is 40.

    Returns:
        np.ndarray: The attenuated and noisy signal, reshaped as a column vector (two-dimensional array with shape (N, 1)).

    Notes:
        - The function applies an exponential attenuation in the frequency domain, modeling a channel that attenuates higher frequencies more.
        - AWGN noise is added using the specified SNR, utilizing the `awgn` helper function.
        - The final output is the real part of the received signal, as imaginary components are discarded.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(1000)
        >>> received_signal = signal_attenuation(data, SNR=30, Fs=1000, Factor=50)
        >>> print(received_signal.shape)
        (1000, 1)

    """
    # Generate frequency indices
    n = np.arange(1, int(len(data) / 2) + 1)
    # Calculate frequency resolution
    fs = 2 * Fs / len(data)
    fs = fs * n
    # Create the attenuation profile for positive frequencies
    ch1 = np.exp(-fs / Factor)
    # Mirror the attenuation profile for negative frequencies
    ch2 = ch1[::-1]
    # Combine to create the full attenuation profile
    ch = np.concatenate([ch1, ch2])

    # Apply attenuation in the frequency domain
    data_xindao_fft = fft(data)
    data_xindao_fft = data_xindao_fft * ch
    # Transform back to the time domain
    datarx1 = ifft(data_xindao_fft)
    # Add AWGN noise to achieve the specified SNR
    datarx1 = awgn(datarx1, SNR)
    # Reshape to a column vector
    datarx1 = datarx1.reshape(-1, 1)
    # Discard any imaginary part introduced by numerical errors
    datarx1 = np.real(datarx1)
    return datarx1


################################################################################
###########                   Main Functions                  ##################
################################################################################
def generate_seq(
        order:          int,
        cons:           np.ndarray,
        num_symbols:    int,
        embedded:       bool=False,
        file_embedded:  str="",
        up_sample:      int=4,
        seed:           int=1,
        taps:           int=35,
        preamble_length:int=32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generates a sequence of random modulated signals (non-attenuated).

    This function creates a sequence of Quadrature Amplitude Modulation (QAM) signals.
    It supports embedding a file into the signal, adding a synchronization preamble,
    and appending an end-of-transmission (EOT) sequence. The generated signal undergoes
    pulse shaping and upsampling to simulate a realistic transmission scenario.

    Parameters:
        order (int): The order of the QAM modulation (e.g., 16 for 16-QAM).
                     Must be a power of 2.
        constellation (np.ndarray): Constellation points.
                                    Defines the mapping between symbols and signal points.
        num_symbols (int): Total number of symbols to generate in the sequence.
        embedded (bool, optional): If True, embeds a file into the signal sequence.
                                   Default is False.
        file_embedded (str, optional): Path to the file to be embedded.
                                       Required if `embedded` is True.
        up_sample (int, optional): Upsampling factor for the signal.
                                   Determines how many samples per symbol are generated.
                                   Default is 4.
        seed (int, optional): Seed for the random number generator to ensure reproducibility.
                              Default is 2.
        taps (int, optional): Filter length. Default is 35.
        preamble_length (int, optional): Length of the synchronization preamble sequence.
                                         Default is 32.

    Returns:
        tuple:
            - final_data (np.ndarray): The final modulated signal after pulse shaping
                                       and attenuation.
            - output_data (np.ndarray): The signal sequence before attenuation.
            - data_afterUpsample (np.ndarray)
            - rand_qamdata (np.ndarray)
            - preamble_upsampled (np.ndarray): The upsampled synchronization preamble.
            - rand_insertion_index (int): The index at which the embedded data starts
                                          within the random QAM data sequence.

    Raises:
        ValueError: If the QAM order is not a power of 2.
        Exception: If the embedded file is not found when `embedded` is True.

    Notes:
        - The function sets the random seed for reproducibility.
        - If embedding a file, the binary data is padded to fit into the QAM symbols.
        - A synchronization preamble and an EOT sequence are added to help in signal
          detection and processing at the receiver end.
        - Pulse shaping is applied using a custom filter to simulate realistic
          transmission conditions.
        - The function ensures that the final signal length matches the specified
          `num_symbols`.

    Example:
        ```python
        final_data, output_data, preamble_upsampled, insertion_index = generate_seq(
            order=16,
            constellation='constellation.txt',
            num_symbols=1000,
            embedded=True,
            file_embedded='data.bin',
            up_sample=8,
            seed=42,
            preamble_length=64
        )
        ```
    """
    np.random.seed(seed)

    if np.log2(order) % 1 != 0:
        raise ValueError("QAM order must be a power of 2.")

    # (1) First read from the constellation file to ensure QAM order is valid

    file_symbols: np.ndarray = []
    # (2) Then process the embedded file, it required
    if embedded:
        # Step 1: Read file and convert to binary
        try:
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            current_dir = os.getcwd()
            file_path = path.join(current_dir, file_embedded)
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
            # print(f"Selected encoded file insertion index: {rand_insertion_index}")
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

    filter_Q_path, filter_I_path = get_filter_paths(num_symbols)

    # Pulse shaping

    gtI1 = filter_I_path.flatten()
    gtQ1 = filter_Q_path.flatten()
    Idata1 = np.real(data_afterUpsample).flatten()
    Qdata1 = np.imag(data_afterUpsample).flatten()

    center_index = int(num_symbols * up_sample / 2 - (taps - 1) / 2)
    filterI = gtI1[center_index: center_index + taps]
    filterQ = gtQ1[center_index: center_index + taps]

    # Convolve data with filters
    DataCapI = np.convolve(Idata1, filterI, mode='same')
    DataCapQ = np.convolve(Qdata1, filterQ, mode='same')

    output_data = DataCapI - DataCapQ
    final_data = signal_attenuation(output_data.flatten())

    return [
        final_data, 
        output_data,
        data_afterUpsample,
        rand_qamdata,
        preamble_upsampled,
        rand_insertion_index
    ]

def demodulation(
    data:           np.ndarray,
    num_symbols:    int,
    constellation:  np.ndarray,
    qam_order:      int=64,
    up_sample:      int=4,
    taps:           int=35,
):
    """
    Perform CAP-QAM demodulation with filter generation included.

    Parameters:
        received_data (np.ndarray): The received signal after channel effects.
        num_symbols (int): Number of symbols in the original sequence.
        up_sample (int): The upsampling factor used during modulation.
        taps (int): Filter length used during modulation.
        constellation (np.ndarray): QAM constellation points.
        alpha (float): Roll-off factor for shaping filter. Default is 0.205.

    Returns:
        np.ndarray: Demodulated QAM symbol indices.
    """
    filter_Q_path, filter_I_path = get_filter_paths(num_symbols)

    # Generate shaping filters
    gt_I = filter_I_path.flatten()
    gt_Q = filter_Q_path.flatten()

    N = len(data)

    # Note: Either method works
    center_index = int(N/2 - (taps - 1)/2)
    # center_index = int(num_symbols * up_sample / 2 - (taps - 1) / 2)
    filterI = gt_I[center_index: center_index + taps]
    filterQ = gt_Q[center_index: center_index + taps]

    # data = downsample(data, up_sample)
    DataCapI = np.convolve(data.flatten(), filterI, mode="same")
    DataCapQ = np.convolve(data.flatten(), filterQ, mode="same")

    # # Combine the I and Q components
    # DataCap = DataCapI - DataCapQ

    DataCap = DataCapI + 1j*DataCapQ
    downsampled = downsample(DataCap, up_sample)

    a = np.linspace(0, qam_order - 1, qam_order, dtype=np.int8)
    b = QAM_mod_from_cons(a, constellation)
    avp = np.sqrt(np.mean(np.abs(b) ** 2))
    recoverdata = downsampled
    recoverdata = recoverdata / np.sqrt(np.mean(np.abs(recoverdata) ** 2)) * avp
    recoverdata = recoverdata.reshape(1, -1)
 
    recoverdata = recoverdata.flatten()
    # # Demodulation: Map to the nearest constellation point
    demodulated_symbols = np.zeros(len(recoverdata), dtype=int)
    for i, sample in enumerate(recoverdata):
        distances = np.abs(sample - constellation)
        demodulated_symbols[i] = np.argmin(distances)  # Find the nearest constellation point

    return demodulated_symbols, recoverdata


################################################################################
###########                   Visualization                   ##################
################################################################################

def graph_IQ_constellation(
        title:          str,
        data:           np.ndarray,
        constellation:  np.ndarray
    ):
    # Create a figure with two subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns

    # Adjust layout to avoid overlap
    
    # ax[0].figure(figsize=(8, 8))
    ax[0].scatter(np.real(data).flatten(), np.imag(data).flatten(), alpha=0.3, color='blue', s=1, label='Received Signal')
    ax[0].scatter(np.real(constellation), np.imag(constellation), alpha=1, color='red', marker='x', s=30, label='Constellation Points')
    ax[0].set_title('Scatter')
    # ax[0].xlabel('In-phase Component (I)')
    # ax[0].ylabel('Quadrature Component (Q)')
    ax[0].axis('square')
    # plt.legend()

    # ax[1].figure(figsize=(8, 8))
    ax[1].hexbin(np.real(data).flatten(), np.imag(data).flatten(), cmap="Blues", label='Received Signal')
    ax[1].scatter(np.real(constellation), np.imag(constellation), alpha=0.7, color='red', marker='x', s=30, label='Constellation Points')
    ax[1].set_title('Histogram')
    # ax[1].xlabel('In-phase Component (I)')
    # ax[1].ylabel('Quadrature Component (Q)')
    ax[1].axis('square')
    # plt.legend()

    for axis in ax.flat:
     axis.set(xlabel='In-phase Component (I)', ylabel='Quadrature Component (Q)')

    plt.suptitle(f'{title} on Constellation Diagram')
    plt.tight_layout()
    plt.show()


# TODO

################################################################################
###########                    STATISTICS!                    ##################
################################################################################

def calculate_ser(rxdata_dec: np.ndarray, DECdata: np.ndarray, taps_LMS: int):
    """
    Calculate Symbol Error Rate (SER).
    
    Parameters:
    - rxdata_dec (ndarray): Received decoded data (symbols)
    - DECdata (ndarray): Reference decoded data (symbols)
    - taps_LMS (int): Number of symbols to skip at the start and end
    
    Returns:
    - symnum (int): Total number of incorrect symbols
    - ser (float): Symbol Error Rate
    """
    rx_symbols = rxdata_dec[taps_LMS:-taps_LMS]
    ref_symbols = DECdata[taps_LMS:-taps_LMS]
    symnum = np.sum(rx_symbols != ref_symbols)
    ser = symnum / len(ref_symbols)
    return symnum, ser

# def calculate_ber(rxdata_dec: np.ndarray, DECdata: np.ndarray, taps_LMS: int):
#     """
#     Calculate Bit Error Rate (BER).
    
#     Parameters:
#     - rxdata_dec (ndarray): Received decoded data (bits)
#     - DECdata (ndarray): Reference decoded data (bits)
#     - taps_LMS (int): Number of bits to skip at the start and end
    
#     Returns:
#     - bitnum (int): Total number of incorrect bits
#     - ber (float): Bit Error Rate
#     """
#     rx_bits = rxdata_dec[taps_LMS:-taps_LMS]
#     ref_bits = DECdata[taps_LMS:-taps_LMS]
#     bitnum = np.sum(rx_bits != ref_bits)
#     ber = bitnum / len(ref_bits)
#     return bitnum, ber

def symbols_to_bits(symbols: np.ndarray, modulation_order: int):
    """
    Convert symbols to binary bits.
    
    Parameters:
    - symbols (ndarray): Array of symbols
    - modulation_order (int): Modulation order (e.g., 2 for QPSK, 4 for 16-QAM)
    
    Returns:
    - bits (ndarray): Flattened array of binary bits
    """
    num_bits_per_symbol = int(np.log2(modulation_order))
    return np.array(
        [int(b) for symbol in symbols for b in format(symbol, f'0{num_bits_per_symbol}b')]
    )

def calculate_ber(rxdata_dec: np.ndarray, DECdata: np.ndarray, taps_LMS: int, modulation_order: int):
    """
    Calculate Bit Error Rate (BER) from symbols.
    
    Parameters:
    - rxdata_dec (ndarray): Received decoded data (symbols)
    - DECdata (ndarray): Reference decoded data (symbols)
    - taps_LMS (int): Number of symbols to skip at the start and end
    - modulation_order (int): Modulation order (e.g., 2 for QPSK, 4 for 16-QAM)
    
    Returns:
    - bitnum (int): Total number of incorrect bits
    - ber (float): Bit Error Rate
    """
    # Trim symbols to remove taps
    rx_symbols = rxdata_dec[taps_LMS:-taps_LMS]
    ref_symbols = DECdata[taps_LMS:-taps_LMS]

    assert len(rx_symbols) == len(ref_symbols)

    rx_bits = symbols_to_bits(rx_symbols, modulation_order)
    ref_bits = symbols_to_bits(ref_symbols, modulation_order)
    
    # Calculate number of bit errors
    bitnum = np.sum(rx_bits != ref_bits)
    
    # Calculate BER
    ber = bitnum / len(ref_bits)
    
    return bitnum, ber

################################################################################
###########                    DEMO DRIVER                    ##################
################################################################################
QAMorder = 64
constellation = 'QAM'
numofsymbols = 51200

def generate_random_seq_with_file_embedded():
    attenurated_data, og_data, preamble_data, insertion_index = generate_seq(
        order=QAMorder,
        constellation=constellation,
        embedded=True,
        file_embedded="test_data.txt",
        num_symbols=numofsymbols
    )

    dataout = attenurated_data / np.sqrt(np.mean(np.abs(attenurated_data) ** 2))
    dataout = dataout.reshape(-1, 1)
    # dataout = signal_attenuation(dataout.flatten())
    dataout_flat = dataout.flatten()

    # awg_SampleRate = 1e9  # Adjust as needed
    # xfre = ((np.arange(len(dataout_flat)) - len(dataout_flat) / 2) * awg_SampleRate) / len(dataout_flat)
    # # DataCap_f = fftshift(fft(dataout_flat))
    # DataCap_f = fft(dataout_flat)
    # fft_signal = fftshift(DataCap_f)

    # plt.figure(figsize=(10, 6))
    # plt.plot(xfre, 20 * np.log10(np.abs(fft_signal) + 1e-12), color="blue")
    # # Limit x-axis and y-axis to positive values only
    # plt.xlim(left=0)  # Set x-axis to start from 0 (positive frequencies only)
    # plt.ylim(bottom=0)  # Set y-axis to start from 0 (positive amplitude only)
    # plt.title(f'Overlap of Two FFT Signals and Region of Difference\n')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (dB)')
    # plt.grid(True)
    # plt.show()

def generate_random_seq():
    attenurated_data, og_data, preamble_data, insertion_index = generate_seq(
        order=QAMorder,
        constellation=constellation,
        num_symbols=numofsymbols
    )

    og_data_flat = og_data.flatten()
    dataout_flat = attenurated_data.flatten()

    from ..visual.fft import graph_fft_overlap

    fft_signal_og = fftshift(fft(og_data_flat))
    fft_signal_rx = fftshift(fft(dataout_flat))
    awg_SampleRate = 1e9  # Adjust as needed
    xfre = ((np.arange(len(dataout_flat)) - len(dataout_flat) / 2) * awg_SampleRate) / len(dataout_flat)
    graph_fft_overlap(
        xfre,
        fft_signal_og,
        xfre,
        fft_signal_rx
    )

    # Plot Frequency Response
    awg_SampleRate = 1e9  # Adjust as needed
    xfre = ((np.arange(len(dataout_flat)) - len(dataout_flat) / 2) * awg_SampleRate) / len(dataout_flat)

    # DataCap_f = fftshift(fft(dataout_flat))
    

    np.savetxt("QAM_test.txt", attenurated_data, delimiter="\n")
    np.savetxt('QAM_target.txt', og_data, delimiter='\n')

    # plt.figure(figsize=(10, 6))
    # # Limit x-axis and y-axis to positive values only
    # plt.plot(xfre, 20 * np.log10(np.abs(fft_signal) + 1e-12), color="blue")
    # plt.xlim(left=0)  # Set x-axis to start from 0 (positive frequencies only)
    # plt.ylim(bottom=0)  # Set y-axis to start from 0 (positive amplitude only)
    # plt.title(f'Overlap of Two FFT Signals and Region of Difference\n')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (dB)')
    # plt.grid(True)
    # plt.show()
    
    data = np.fromfile("../../src/pred.txt", sep="\n")
    print("data", len(data), data[:10])

    cons = read_constellation_file(QAMorder, constellation)
    demodulation(
        attenurated_data,
        # data,
        numofsymbols,
        cons,
    )

    # Step 3: Demodulate the Signal
def qam_demodulate(received_signal, cons):
    distances = np.abs(received_signal.reshape(-1, 1) - cons.reshape(1, -1))
    estimated_symbols = np.argmin(distances, axis=1)
    return estimated_symbols


if __name__ == "__main__":
    

    generate_random_seq()
    # generate_random_seq_with_file_embedded()

    # cons = read_constellation_file(QAMorder, constellation)
    # cap_qam_demodulation_with_filter_generation(
    #     attenurated_data,
    #     filter_I_path,
    #     filter_Q_path,
    #     cons,
    #     numofsymbols
    # )

    pass
# %%
