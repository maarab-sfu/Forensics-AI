import reedsolo
import random
import torch
import config as c

def byte_to_bits(the_bytes):
    return ''.join(format(byte, '08b') for byte in the_bytes)

def bits_to_bytearray(bits):
    # Ensure the length of the binary string is a multiple of 8
    if len(bits) % 8 != 0:
        raise ValueError("The length of the binary string must be a multiple of 8")
    # Convert bits to a bytearray
    return bytearray(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

def encode_message(message_bits, ecc):
    # Pad the message to ensure it's divisible by 8 (if needed)
    if len(message_bits) % 8 != 0:
        padding_length = 8 - (len(message_bits) % 8)
        message_bits += [0] * padding_length  # Pad with zeros
    # Message is assumed to be a list of integers (0 or 1)
    message_bytes = [int("".join(map(str, message_bits[i:i+8])), 2) for i in range(0, len(message_bits), 8)]
    
    
    # Create Reed-Solomon codec with n = 100 and m = 64
    rs = reedsolo.RSCodec(ecc)  # Calculate redundancy based on message size
    encoded_message = rs.encode(bytes(message_bytes))
    
    return encoded_message

def decode_message(encoded_message, ecc):
    rs = reedsolo.RSCodec(ecc)  # Create Reed-Solomon codec
    try:
        decoded_bytes = rs.decode(encoded_message)[0]
        return byte_to_bits(decoded_bytes)
    except reedsolo.ReedSolomonError as e:
        # Log the error to the console
        print(f"Reed-Solomon decoding failed: {e}")
        return None

def generate_random_bits(bit_length):
    return [random.choice([0, 1]) for _ in range(bit_length)]

def string_to_tensor(bit_string):
    # Convert the bit string to a list of integers (0 or 1)
    bit_list = [int(bit) for bit in bit_string]
    
    # Convert the list of integers to a PyTorch tensor
    tensor = torch.tensor(bit_list, dtype=torch.float32)  # You can change dtype to torch.float32 or others if needed
    
    return tensor

def tensor_to_string(tensor):
    # Convert the tensor to a list of integers (0 or 1)
    bit_list = tensor.tolist()  # Convert tensor to a list
    # Convert the list of integers to a string of '0's and '1's
    bit_string = ''.join(str(bit) for bit in bit_list)
    
    return bit_string

def print_comparison(str1, str2):
    # Determine the max length and pad the shorter string with spaces
    max_length = max(len(str1), len(str2))
    str1 = str1.ljust(max_length)
    str2 = str2.ljust(max_length)

    # Function to add the byte indicator '|' every 8 bits
    def format_with_byte_indicator(bit_string):
        return '|'.join([bit_string[i:i+8] for i in range(0, len(bit_string), 8)])

    # Format both strings with byte indicators
    formatted_str1 = format_with_byte_indicator(str1)
    formatted_str2 = format_with_byte_indicator(str2)

    # Print both strings with byte indicators
    print("encoded:     ", formatted_str1)
    print("extracted:   ", formatted_str2)

    # Generate difference indicator
    indicator = ''.join('^' if i < min(len(str1.strip()), len(str2.strip())) and str1[i] != str2[i] 
                        else '+' if i >= min(len(str1.strip()), len(str2.strip())) else ' ' 
                        for i in range(max_length))
    
    # Insert the byte indicator for differences
    formatted_indicator = format_with_byte_indicator(indicator)
    print("Difference:  ", formatted_indicator)

def flip_bit(bit_string, position):
    # Convert the string to a list so we can modify it (strings are immutable in Python)
    bit_list = list(bit_string)
    
    # Flip the bit at the specified position
    if bit_list[position] == '0':
        bit_list[position] = '1'
    else:
        bit_list[position] = '0'
    
    # Convert the list back to a string
    return ''.join(bit_list)

def main():
    # Example usage:
    # Generate a random message
    n = c.wm_cap
    m = c.hash_length * c.hash_length

    message_bits = generate_random_bits(m)

    # Pad the message to ensure it's divisible by 8 (if needed)
    if len(message_bits) % 8 != 0:
        padding_length = 8 - (len(message_bits) % 8)
        message_bits += [0] * padding_length  # Pad with zeros

    m = len(message_bits)

    ecc = (n-m)//8
    print("There are {} ecc symbols.".format(ecc))

    # Encode the message
    encoded = encode_message(message_bits, ecc)
    encoded_bits = byte_to_bits(encoded)


    # Simulate a bit flip (introducing errors in the encoded message)
    import random
    encoded_with_errors = bytearray(encoded)
    encoded_with_errors_bits = byte_to_bits(encoded_with_errors)

    counter = 0
    num_of_flip = random.randint(2, 6)
    print("{} bits flipped!".format(num_of_flip))
    for i in range(num_of_flip):  # Randomly flip 2 to 8 bits
        pos = random.randint(0, len(encoded_with_errors_bits) - 1)
        encoded_with_errors_bits = flip_bit(encoded_with_errors_bits, pos)
        # print("The bit in position {} is flipped from {} to {}".format(pos, encoded_with_errors_bits[pos], new_bit))
        # encoded_with_errors_bits[pos] = new_bit  # Flip the bit at the chosen position
        counter += 1

    print_comparison(encoded_bits, encoded_with_errors_bits)
    print("\n")
    # print("Encoded Message (in bits):          ", encoded_bits)
    # print("Encoded Message with error(in bits):", byte_to_bits(encoded_with_errors))

    print("Encoded Message length:", len(encoded_bits))

    print("{} bits flipped!".format(counter))

    # Decode the message
    decoded_message = decode_message(encoded_with_errors, ecc)
    # print(type(decoded_message))

    # Display the original and decoded message
    print("Original Message (in bits):", ''.join(map(str, message_bits)))
    print("Original Message len:", len(''.join(map(str, message_bits))))
    print("Decoded Message (in bits): ", decoded_message)
    print("Decoded Message length:", len(decoded_message))

if __name__ == '__main__':
    main()
