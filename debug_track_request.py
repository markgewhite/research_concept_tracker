#!/usr/bin/env python3
"""
Debug script to test the /api/track endpoint
Run this with the debugger while the FastAPI server is running
"""

import httpx
import json


def main():
    """Send a tracking request to the API"""

    url = "http://localhost:8000/api/track"

    payload = {
        "seed_paper_ids": ["1706.03762"],
        "end_date": "2019-12-31",
        "window_months": 6,
        "max_papers_per_window": 500
    }

    print(f"Sending POST request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)

    # You can set a breakpoint on the next line to inspect the payload
    response = httpx.post(url, json=payload, timeout=300.0)

    # Set a breakpoint here to inspect the response
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nTracking completed successfully!")
        print(f"Number of steps: {result['num_steps']}")
        print(f"Total papers: {result['total_papers']}")
        print(f"Date range: {result['date_range']}")

        # Print details of each step
        for step in result['timeline']:
            print(f"\n{'='*80}")
            print(f"Step {step['step_number']}: {step['start_date']} to {step['end_date']}")
            print(f"  Papers: {len(step['papers'])}")
            print(f"  Avg Similarity: {step['avg_similarity']:.3f}")
            print(f"  Confidence: High={step['num_high_confidence']}, "
                  f"Moderate={step['num_moderate']}, Low={step['num_low']}")

            # Show top 20 papers by similarity
            papers_sorted = sorted(step['papers'], key=lambda p: p['similarity'], reverse=True)
            print(f"\n  Top 20 papers:")
            for i, paper in enumerate(papers_sorted[:20], 1):
                sim = paper.get('similarity', 0)
                print(f"\n    {i}. [{paper['arxiv_id']}] sim={sim:.3f}")
                print(f"       Title: {paper['title']}")
                if i <= 10:  # Show full abstract for top 10
                    print(f"       Abstract: {paper['abstract']}")
                    print(f"       Categories: {', '.join(paper.get('categories', []))}")
    else:
        print(f"\nError: {response.text}")

    return response


if __name__ == "__main__":
    main()
