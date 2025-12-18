import math
import random
import numpy as np
from python_polar_coding.polar_codes import FastSSCPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message


class SimpleBSCChannel:
    def __init__(self, error_rate):
        self.error_rate = error_rate
        self.llr_0 = math.log((1 - error_rate) / error_rate)
        self.llr_1 = math.log(error_rate / (1 - error_rate))

    def transmit(self, message):
        llrs = []
        for bit in message:
            if random.random() < self.error_rate:
                received_bit = 1 - bit  # Flip the bit
            else:
                received_bit = bit
            # llrs.append(self.llr_0 if received_bit == 0 else self.llr_1)
            llrs.append(received_bit)
        return np.array(llrs)


def calculate_code_parameters(bit_accuracy, total_length=256):
    if bit_accuracy > 1:
        bit_accuracy /= 100  # Convert percentage to fraction
    error_rate = 1 - bit_accuracy

    # Calculate parameters using heuristic similar to BCH approach
    m = int(math.log2(total_length))
    t = math.ceil(total_length * error_rate)
    redundancy = m * t
    K = total_length - redundancy

    # Ensure valid code parameters
    K = max(8, min(K, total_length - 8))  # Keep within reasonable bounds
    return K, error_rate


# def test_polar_code(bit_accuracy=0.91):
#     # Fixed parameters
#     N = 256  # Total length (must be power of 2)
#     design_snr = 0.0  # Standard design SNR

#     # Calculate code parameters
#     K, error_rate = calculate_code_parameters(bit_accuracy)

#     # Initialize components
#     codec1 = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
#     codec2 = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
#     # error_rate = 0.28999999999999997
#     print("Error rate: ", error_rate)
#     bsc = SimpleBSCChannel(error_rate=error_rate)

#     # Generate test message
#     # TODO: change this line and assign the desired 64bits message to msg variable.
#     msg = generate_binary_message(size=K)

#     # Encoding
#     encoded = codec1.encode(msg)
    

#     # Transmission through BSC
#     # TODO: the received_llrs variable, should be the extracted noisy watermark. so, ignore
#     # TODO: the bsc.transmit function. it supposed to simulate the noise for test only.
#     received_llrs = bsc.transmit(encoded)
#     # received_llrs = generate_binary_message(size=N)
#     print(received_llrs)

#     # Decoding
#     # TODO: read the decoded value as an error-free message.
#     decoded = codec2.decode(received_llrs)
#     print(decoded)

#     # Results
#     success = np.array_equal(msg, decoded)

#     print(f"\n=== Polar Code Test @ {bit_accuracy * 100:.1f}% bit accuracy ===")
#     print(f"Code parameters: (N={N}, K={K}), Redundancy={N - K}")
#     print(f"Message length: {len(msg)} bits")
#     print("\nOriginal message (first 20 bits):", ''.join(map(str, msg[:20])))
#     print("Encoded message (first 20 bits):", ''.join(map(str, encoded[:20])))
#     print("\nDecoded successfully!" if success else "\nDecoding failed!")
#     return success, msg, encoded, received_llrs, decoded

def initialize_polar(N = 256, K = 64):
    # Initialize components
    codec = FastSSCPolarCodec(N=N, K=K, design_snr=0)
    return codec

def embed_polar(codec, msg):
    encoded = codec.encode(msg)
    return encoded
def extract_polar(codec, encoded):
    decoded = codec.decode(encoded)
    return decoded

if __name__ == "__main__":
    codec = initialize_polar(256, 64)
    error_rate=0.2

    for i in range(10):
        msg = generate_binary_message(size=64)
        
        enc = embed_polar(codec, msg)
        
        
        
        random_number = random.uniform(error_rate, 2 * error_rate)
        print("The random error_rate is: ", random_number)
        bsc = SimpleBSCChannel(random_number)
        received_llrs = bsc.transmit(enc)

        llr_0 = math.log((1 - error_rate) / error_rate)
        llr_1 = math.log(error_rate / (1 - error_rate))

        firm_received_llrs = np.where(received_llrs == 0, llr_0, llr_1)
        # received_llrs = (received_llrs>0).astype(np.int32)
        
        ex_msg = extract_polar(codec, firm_received_llrs)
        
        match_percentage = np.mean(msg == ex_msg) * 100

        print("acc: ", match_percentage)

        match_percentage = np.mean(enc == received_llrs) * 100


        # print("message: ", msg)
        # print("encoded: ", enc)
        # print("noised: ", (received_llrs>0).astype(np.int32))
        # print("extracted: ", ex_msg)
        # print("acc: ", match_percentage)
        # print("noise acc: ", match_percentage)
    
    
    
    
    
    
    
    
    
    
    
    
    # # Example test with 96% bit accuracy
    # success, original, encoded, received, decoded = test_polar_code(bit_accuracy=0.91)

    # received = binary_array = (received < 0).astype(int)
    # print('number of total bit flips:', len(np.where(encoded != received)[0]))
    # print('encoded vs received', np.where(encoded != received)[0])
    # # print('original vs decoded', original == decoded)
    # # For detailed binary comparison
    # errors = np.where(original != decoded)[0]
    # print(f"Number of bit errors: {len(errors)}")
    # print(f"original vs decoded: {errors[:10]}" if len(errors) > 0 else "No errors")
    # # if not success:
    # #     # print("\nDetailed comparison:")
    # #     errors = np.where(original != decoded)[0]
    # #     print(f"Number of bit errors: {len(errors)}")
    # #     print(f"original vs decoded: {errors[:10]}" if len(errors) > 0 else "No errors")
