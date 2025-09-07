#!/usr/bin/env fish

# Process the first 20 images from demo_entity directory
# and generate HTML reports with matching names

set con_out (conda activate SecureFace)

echo $con_out

# Create results directory if it doesn't exist
mkdir -p results

set count 0
for image in demo_entity/*.jpg
    # Break after processing 20 images
    if test $count -ge 20
        break
    end

    # Get the base name without extension
    set base_name (basename $image .jpg)

    # Set output HTML file name
    set output_html "results/$base_name.html"

    # Run the search script
    echo "Processing $image -> $output_html"
    python search_face_matches.py $image -o $output_html

    # Increment counter
    set count (math $count + 1)
end

echo "Processed $count images"
