import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np

def analyze_lfw_dataset(dataset_path):
    # Step 1: Count the number of images per person
    person_img_count = {}
    
    # Get list of all person directories
    person_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Count images for each person
    for person in person_dirs:
        person_path = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith('.jpg')]
        person_img_count[person] = len(image_files)
    
    # Step 2: Bar chart for top 20 people with most images
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(list(person_img_count.items()), columns=['Person', 'Image Count'])
    df = df.sort_values('Image Count', ascending=False)
    
    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Image Count', y='Person', hue='Person', data=df.head(20), legend=False)
    plt.title('Top 20 People with Most Images in LFW Dataset')
    plt.tight_layout()
    plt.savefig('top20_people.png')
    plt.close()
    
    # Step 3: Pie chart showing distribution
    # Group people by image count
    count_bins = {
        '1 image': 0,
        '2-4 images': 0,
        '5-9 images': 0,
        '10-49 images': 0,
        '50+ images': 0
    }
    
    for count in person_img_count.values():
        if count == 1:
            count_bins['1 image'] += 1
        elif 2 <= count <= 4:
            count_bins['2-4 images'] += 1
        elif 5 <= count <= 9:
            count_bins['5-9 images'] += 1
        elif 10 <= count <= 49:
            count_bins['10-49 images'] += 1
        else:  # count >= 50
            count_bins['50+ images'] += 1
    
    # Plot the pie chart with improved label placement
    plt.figure(figsize=(13, 11), facecolor='white') # Increased figsize
    colors = sns.color_palette('viridis', len(count_bins))
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(
        count_bins.values(),
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        explode=[0.05, 0.05, 0.05, 0.05, 0.05],
        colors=colors,
        pctdistance=0.85,
        wedgeprops={'edgecolor': 'w', 'linewidth': 2},
        textprops={'visible': False}  # Hide default labels
    )
    
    # Configure the percentage labels inside pie slices
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # Define custom positions for each label to prevent overlap
    label_positions = [
        # Format: (horizontal_position, vertical_position)
        (-1.3, -0.5),    # 1 image (large segment at bottom-left)
        (1.3, 0.3),      # 2-4 images (medium segment at right side)
        (0.9, 1.1),      # 5-9 images (small segment at top-right)
        (0.0, 1.3),     # 10-49 images (small segment at top-center, highest)
        (-0.9, 1.1)      # 50+ images (smallest segment at top-left)
    ]
    
    # Add labels with custom connecting lines
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y_coord = np.sin(np.deg2rad(ang))
        x_coord = np.cos(np.deg2rad(ang))
        
        # Start point of line on the pie edge
        xy = (x_coord, y_coord) # Start line from edge of wedge
        
        # Custom end point for the label
        xytext = label_positions[i]
        
        # Style for the annotation box
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=0.72)
        
        # Draw the connecting line and label
        plt.annotate(
            list(count_bins.keys())[i], 
            xy=xy, 
            xytext=xytext,
            ha='center',
            va='center',
            bbox=bbox_props,
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3,rad=0",
                color='black',
                lw=1
            )
        )
    
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.title('Distribution of People by Number of Images', fontsize=14, pad=20)
    
    # Remove unnecessary axis elements
    plt.gca().set_frame_on(False)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
    # Add some padding around the figure
    plt.tight_layout(pad=2.5) # Increased padding
    plt.savefig('image_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Print summary statistics
    total_people = len(person_img_count)
    total_images = sum(person_img_count.values())
    
    print(f"Dataset Summary:")
    print(f"Total number of people: {total_people}")
    print(f"Total number of images: {total_images}")
    print(f"Average images per person: {total_images / total_people:.2f}")
    print("\nDistribution of people by image count:")
    for bin_name, bin_count in count_bins.items():
        print(f"  {bin_name}: {bin_count} people ({bin_count/total_people*100:.1f}%)")
    
    return df, count_bins

if __name__ == "__main__":
    # The dataset path should be adjusted according to the actual location
    dataset_path = "lfwver2/lfw-deepfunneled/lfw-deepfunneled"
    
    # Analyze the dataset
    top_people_df, distribution = analyze_lfw_dataset(dataset_path)
    
    # Display the top 5 people with most images
    print("\nTop 5 people with most images:")
    for idx, row in top_people_df.head(5).iterrows():
        print(f"  {row['Person']}: {row['Image Count']} images") 