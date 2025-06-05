# Example Usage of DTW Pipeline with Your Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the DTW pipeline components
from dtw_pipeline_complete import extract_and_normalize_pitch, process_metadata_csv, DTWAnalyzer

def demo_dtw_pipeline():
    """
    Demonstration of how to use the DTW pipeline with your metadata.csv
    """
    
    print("DTW Pipeline Demo")
    print("="*50)
    
    # Step 1: Extract pitch data directly from your audio files
    print("Step 1: Extracting normalized pitch contours from audio files...")
    
    try:
        # Use your existing metadata.csv file
        pitch_data = process_metadata_csv('metadata.csv', normalization_method='semitones')
        print(f"✓ Successfully extracted data for {len(pitch_data)} pairs")
        
        # Show what data structure looks like
        if pitch_data:
            first_pair = list(pitch_data.keys())[0]
            print(f"\nData structure for {first_pair}:")
            print(f"  Student file: {pitch_data[first_pair]['student']['file']}")
            print(f"  Teacher file: {pitch_data[first_pair]['teacher']['file']}")
            print(f"  Student pitch contour length: {len(pitch_data[first_pair]['student']['pitch'])}")
            print(f"  Teacher pitch contour length: {len(pitch_data[first_pair]['teacher']['pitch'])}")
            print(f"  Student voicing: {pitch_data[first_pair]['student']['stats']['voicing_percentage']:.1f}%")
            print(f"  Teacher voicing: {pitch_data[first_pair]['teacher']['stats']['voicing_percentage']:.1f}%")
        
    except Exception as e:
        print(f"✗ Error extracting pitch data: {e}")
        print("Make sure your metadata.csv and audio files are in the correct location")
        return
    
    # Step 2: Initialize DTW analyzer
    print(f"\nStep 2: Initializing DTW analyzer...")
    dtw_analyzer = DTWAnalyzer(pitch_data)
    print("✓ DTW analyzer initialized")
    
    # Step 3: Analyze a single pair as example
    print(f"\nStep 3: Running DTW analysis on {first_pair}...")
    try:
        result = dtw_analyzer.analyze_single_pair(first_pair)
        
        print(f"✓ DTW analysis completed for {first_pair}")
        print(f"  Total DTW cost: {result['total_dtw_cost']:.4f}")
        print(f"  Optimal path length: {result['path_length']} alignments")
        print(f"  Student duration: {result['student_duration']:.2f} seconds")
        
        # Show cost aggregation results
        print(f"\n  Average Cost Aggregation:")
        avg_costs = result['cost_aggregation']['average']
        for note_pair, cost in list(avg_costs.items())[:5]:  # Show first 5
            print(f"    {note_pair[0]} → {note_pair[1]}: {cost:.4f}")
        if len(avg_costs) > 5:
            print(f"    ... and {len(avg_costs) - 5} more note pairs")
            
        print(f"\n  Maximum Cost Aggregation:")
        max_costs = result['cost_aggregation']['max']
        for note_pair, cost in list(max_costs.items())[:5]:  # Show first 5
            print(f"    {note_pair[0]} → {note_pair[1]}: {cost:.4f}")
        if len(max_costs) > 5:
            print(f"    ... and {len(max_costs) - 5} more note pairs")
        
    except Exception as e:
        print(f"✗ Error in DTW analysis: {e}")
        return
    
    # Step 4: Generate cost matrix visualization
    print(f"\nStep 4: Generating cost matrix visualization...")
    try:
        dtw_analyzer.visualize_cost_matrix(result, save_path=f'demo_cost_matrix_{first_pair}.png')
        print(f"✓ Cost matrix visualization saved as 'demo_cost_matrix_{first_pair}.png'")
    except Exception as e:
        print(f"✗ Error generating visualization: {e}")
    
    # Step 5: Run analysis on all pairs
    print(f"\nStep 5: Running DTW analysis on all pairs...")
    try:
        all_results = dtw_analyzer.run_full_analysis()
        print(f"✓ Analysis completed for {len(all_results)} pairs")
        
        # Generate summary
        total_costs = [r['total_dtw_cost'] for r in all_results.values()]
        print(f"\nSummary Statistics:")
        print(f"  Average DTW cost: {np.mean(total_costs):.4f}")
        print(f"  Min DTW cost: {np.min(total_costs):.4f}")
        print(f"  Max DTW cost: {np.max(total_costs):.4f}")
        print(f"  Std DTW cost: {np.std(total_costs):.4f}")
        
    except Exception as e:
        print(f"✗ Error in full analysis: {e}")
        return
    
    # Step 6: Save results
    print(f"\nStep 6: Saving results to CSV...")
    try:
        # Create detailed results DataFrame
        detailed_results = []
        
        for pair_id, result in all_results.items():
            metadata = pitch_data[pair_id]['metadata']
            
            row = {
                'pair_id': pair_id,
                'student_file': metadata['s_file'],
                'teacher_file': metadata['t_file'],
                'total_dtw_cost': result['total_dtw_cost'],
                'path_length': result['path_length'],
                'student_duration': result['student_duration']
            }
            
            # Add some key note pair costs
            avg_costs = result['cost_aggregation']['average']
            max_costs = result['cost_aggregation']['max']
            
            # Look for perfect matches (Sa→Sa, Re→Re, etc.)
            perfect_matches = [(k, v) for k, v in avg_costs.items() if k[0] == k[1]]
            if perfect_matches:
                perfect_avg = np.mean([v for k, v in perfect_matches])
                perfect_max = np.mean([max_costs[k] for k, v in perfect_matches if k in max_costs])
                row['perfect_match_avg_cost'] = perfect_avg
                row['perfect_match_max_cost'] = perfect_max
            
            detailed_results.append(row)
        
        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv('demo_dtw_results.csv', index=False)
        print(f"✓ Results saved to 'demo_dtw_results.csv'")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
    
    print(f"\n" + "="*50)
    print("Demo completed! You now have:")
    print("  1. Normalized pitch contours extracted from your audio files")
    print("  2. DTW cost matrices computed with log-scale distance")
    print("  3. Optimal DTW paths found for all pairs")
    print("  4. Note correspondences mapped to SARGAM")
    print("  5. Cost aggregations (average and max) for all note pairs")
    print("  6. Visualizations and CSV results saved")

if __name__ == "__main__":
    demo_dtw_pipeline()