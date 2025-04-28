import os
import re
import math
import numpy as np
import csv
from collections import namedtuple

# Constants
CHUNK_SIZE = 4096  # 4KB chunks
MAX_CHUNKS = 3600  # Fixed input size for CNN
SECTION_ORDER = [
    'Header', '.data', '.edata', '.idata', '.pdata', '.rdata',
    '.rsrc', '.reloc', '.text', '.tls', '.sdata', '.xdata', 'Undefined'
]
Interval = namedtuple('Interval', ['start', 'end', 'section_index'])

def compute_entropy(data):
    """Calculate Shannon entropy of a byte chunk."""
    if len(data) == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    non_zero_probs = probs[probs > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    # Set very small (or negative zero) entropy to exactly 0.0
    if np.isclose(entropy, 0.0, atol=1e-12):
        entropy = 0.0
    return entropy

def parse_sections_from_asm(asm_path):
    """Parse section intervals from .asm file."""
    sections = []
    current_section = None
    start_addr = 0

    with open(asm_path, 'r', errors='ignore') as f:
        for line in f:
            # Match section headers (e.g., ".text:00401000")
            section_match = re.match(r'^\.(text|data|rsrc|rdata|idata|edata|pdata|reloc|tls)\:', line)
            if section_match:
                section_name = section_match.group(1)
                hex_addr = line.split(':')[1].strip()[:8]  # Extract virtual address
                start_addr = int(hex_addr, 16)
                sections.append({
                    'name': section_name,
                    'start': start_addr,
                    'end': start_addr  # Temporary, updated with size later
                })
                current_section = section_name
            
            # Match section size (e.g., "Virtual size: 00001000")
            size_match = re.search(r'Virtual size\s*:\s*([0-9A-F]+)', line, re.IGNORECASE)
            if current_section and size_match:
                size = int(size_match.group(1), 16)
                sections[-1]['end'] = sections[-1]['start'] + size
                current_section = None

    return sections

def get_section_intervals(sections, file_size):
    """Convert parsed sections into Interval objects."""
    intervals = []
    
    # Add header interval (assume it's missing in .asm)
    intervals.append(Interval(0, sections[0]['start'] if sections else file_size, 0))
    
    # Add sections
    predefined_sections = {name.lower(): idx+1 for idx, name in enumerate([
        '.data', '.edata', '.idata', '.pdata', '.rdata', '.rsrc', '.reloc',
        '.text', '.tls', '.sdata', '.xdata'
    ])}
    
    for section in sections:
        section_name = f".{section['name'].lower()}"
        section_index = predefined_sections.get(section_name.lower(), 12)  # Undefined
        intervals.append(Interval(section['start'], section['end'], section_index))
    
    # Sort and fill gaps
    sorted_intervals = sorted(intervals, key=lambda x: x.start)
    all_intervals = []
    prev_end = 0
    
    for interval in sorted_intervals:
        if interval.start > prev_end:
            all_intervals.append(Interval(prev_end, interval.start, 12))  # Undefined
        all_intervals.append(interval)
        prev_end = interval.end
    
    if prev_end < file_size:
        all_intervals.append(Interval(prev_end, file_size, 12))  # Undefined
    
    return all_intervals

def bytes_to_binary(bytes_path):
    """Convert .bytes file to raw binary (replacing ?? with FF)."""
    binary_data = bytearray()
    with open(bytes_path, 'r') as f:
        for line in f:
            hex_part = re.sub(r'^[0-9A-F]{8}\s+', '', line.strip())
            hex_bytes = hex_part.split()
            for b in hex_bytes:
                binary_data.append(int(b, 16) if b != '??' else 0xFF)
    return bytes(binary_data)

def generate_feature_vector(bytes_path, asm_path):
    """Generate structural entropy feature vector."""
    # Convert .bytes to binary
    binary_data = bytes_to_binary(bytes_path)
    file_size = len(binary_data)
    
    # Parse sections from .asm
    sections = parse_sections_from_asm(asm_path)
    intervals = get_section_intervals(sections, file_size)
    
    # Process chunks
    feature_vectors = []
    for interval in intervals:
        section_data = binary_data[interval.start:interval.end]
        num_chunks = len(section_data) // CHUNK_SIZE
        remainder = len(section_data) % CHUNK_SIZE
        
        for i in range(num_chunks):
            chunk = section_data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
            entropy = compute_entropy(chunk)
            one_hot = [0.0] * 13
            one_hot[interval.section_index] = 1.0
            feature_vectors.append([entropy] + one_hot)
        
        if remainder > 0:
            chunk = section_data[-remainder:].ljust(CHUNK_SIZE, b'\x00')
            entropy = compute_entropy(chunk)
            one_hot = [0.0] * 13
            one_hot[interval.section_index] = 1.0
            feature_vectors.append([entropy] + one_hot)
    
    # Pad/truncate to MAX_CHUNKS
    if len(feature_vectors) > MAX_CHUNKS:
        feature_vectors = feature_vectors[:MAX_CHUNKS]
    else:
        padding = [[0.0]*14 for _ in range(MAX_CHUNKS - len(feature_vectors))]
        feature_vectors += padding
    
    return np.array(feature_vectors, dtype=np.float32)

def save_to_csv(feature_vector, output_path):
    """Save feature vector to CSV."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(feature_vector)

def process_dataset(input_dir, output_dir):
    """Process all .bytes files in input_dir with matching .asm files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.bytes'):
            continue
        
        base_name = os.path.splitext(filename)[0]
        bytes_path = os.path.join(input_dir, filename)
        asm_path = os.path.join(input_dir, f"{base_name}.asm")
        output_path = os.path.join(output_dir, f"{base_name}.csv")
        
        if not os.path.exists(asm_path):
            print(f"Skipping {filename}: No matching .asm file")
            continue
        
        try:
            features = generate_feature_vector(bytes_path, asm_path)
            save_to_csv(features, output_path)
            print(f"Processed: {filename} -> {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_directory = "Microsoft_dataset/Original/test"  # Contains .bytes and .asm files
    output_directory = "ProcessedData/microsoft_entropy_csv/test"  # Where CSV files will be saved
    
    process_dataset(input_directory, output_directory)