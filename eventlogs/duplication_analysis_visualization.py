import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

def parse_duplication_analysis(file_path):
    """Parse the duplication analysis text file and extract data."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract individual file results
    file_pattern = r'seq(\d+)\.h5:\s*\n\s*Total events: ([\d,]+)\s*\n\s*Unique events: ([\d,]+)\s*\n\s*Duplicate percentage: ([\d.]+)%\s*\n\s*Duplicate events: ([\d,]+)'
    matches = re.findall(file_pattern, content)
    
    data = []
    for match in matches:
        seq_num = int(match[0])
        total_events = int(match[1].replace(',', ''))
        unique_events = int(match[2].replace(',', ''))
        duplicate_percentage = float(match[3])
        duplicate_events = int(match[4].replace(',', ''))
        
        data.append({
            'sequence': seq_num,
            'total_events': total_events,
            'unique_events': unique_events,
            'duplicate_percentage': duplicate_percentage,
            'duplicate_events': duplicate_events
        })
    
    # Sort by sequence number
    data.sort(key=lambda x: x['sequence'])
    
    # Extract per-file statistics
    stats_pattern = r'Average duplicate percentage: ([\d.]+)%\s*\nMedian duplicate percentage: ([\d.]+)%\s*\nMinimum duplicate percentage: ([\d.]+)%\s*\nMaximum duplicate percentage: ([\d.]+)%\s*\nStandard deviation: ([\d.]+)%'
    stats_match = re.search(stats_pattern, content)
    
    stats = {}
    if stats_match:
        stats = {
            'average': float(stats_match.group(1)),
            'median': float(stats_match.group(2)),
            'minimum': float(stats_match.group(3)),
            'maximum': float(stats_match.group(4)),
            'std_dev': float(stats_match.group(5))
        }
    
    return data, stats

def create_visualization(data, stats):
    """Create simplified visualization with only two plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Duplicate Percentage by Sequence
    sequences = [d['sequence'] for d in data]
    duplicate_percentages = [d['duplicate_percentage'] for d in data]
    
    # Color code based on duplicate percentage
    colors = ['red' if p > 10 else 'orange' if p > 5 else 'green' for p in duplicate_percentages]
    
    bars = ax1.bar(sequences, duplicate_percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Sequence Number')
    ax1.set_ylabel('Duplicate Percentage (%)')
    ax1.set_title('Duplicate Percentage by Sequence')
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    ax1.axhline(y=stats['average'], color='blue', linestyle='--', alpha=0.7, label=f'Average: {stats["average"]:.2f}%')
    ax1.axhline(y=stats['median'], color='purple', linestyle='--', alpha=0.7, label=f'Median: {stats["median"]:.2f}%')
    ax1.legend()
    
    # 2. Top 10 Sequences by Duplicate Percentage
    sorted_data = sorted(data, key=lambda x: x['duplicate_percentage'], reverse=True)[:10]
    top_sequences = [d['sequence'] for d in sorted_data]
    top_percentages = [d['duplicate_percentage'] for d in sorted_data]
    
    bars = ax2.barh(range(len(top_sequences)), top_percentages, color='coral', alpha=0.7)
    ax2.set_yticks(range(len(top_sequences)))
    ax2.set_yticklabels([f'seq{d["sequence"]:02d}' for d in sorted_data])
    ax2.set_xlabel('Duplicate Percentage (%)')
    ax2.set_title('Top 10 Sequences by Duplicate Percentage')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars, top_percentages)):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the visualization."""
    # Parse the data
    data, stats = parse_duplication_analysis('duplication_analysis.txt')
    
    # Create visualization
    fig = create_visualization(data, stats)
    
    plt.savefig('duplication_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some key insights
    print("Key Insights from Duplication Analysis:")
    print("=" * 50)
    print(f"• Files with >10% duplicates: {sum(1 for d in data if d['duplicate_percentage'] > 10)}")
    print(f"• Files with <1% duplicates: {sum(1 for d in data if d['duplicate_percentage'] < 1)}")
    print(f"• Highest duplicate rate: seq{max(data, key=lambda x: x['duplicate_percentage'])['sequence']:02d} ({max(data, key=lambda x: x['duplicate_percentage'])['duplicate_percentage']:.2f}%)")
    print(f"• Lowest duplicate rate: seq{min(data, key=lambda x: x['duplicate_percentage'])['sequence']:02d} ({min(data, key=lambda x: x['duplicate_percentage'])['duplicate_percentage']:.2f}%)")

if __name__ == "__main__":
    main()
