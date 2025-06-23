import os
import glob
import argparse

def create_filelist(datadir, output_file):
    # Find all JPEG files (case-sensitive)
    jpeg_files = glob.glob(os.path.join(datadir, '*.JPEG'))
    
    # Convert to relative paths
    rel_paths = [os.path.relpath(f, start=datadir) for f in jpeg_files]
    
    # Sort alphabetically
    rel_paths.sort()
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(rel_paths) + "\n")

if __name__ == "__main__":
    
    output = create_filelist(datadir="data/ILSVRC2012_validation/data",
                             output_file="data/LSVRC2012_validation")
    
    