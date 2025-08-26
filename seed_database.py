#!/usr/bin/env python3
"""
Script to seed the database with performance data for skill level prediction.
"""

from database import Database

def main():
    print("🌱 Seeding database with performance data...")
    
    # Initialize database
    db = Database()
    
    # Seed performance data
    success = db.seed_performance_data()
    
    if success:
        print("✅ Successfully seeded 500 performance records (100 per skill level)")
        
        # Show statistics
        stats = db.get_performance_stats()
        print("\n📊 Performance Data Statistics:")
        print("-" * 50)
        
        for skill_level, data in stats.items():
            print(f"{skill_level.capitalize():>12}: {data['count']:>3} records | "
                  f"Avg Score: {data['avg_score']:>5.1f}% | "
                  f"Avg Time: {data['avg_time']:>5.1f}s")
        
        print("\n🎯 Testing prediction system...")
        
        # Test predictions
        test_cases = [
            (95, 120, "Professional"),
            (85, 250, "Master"),
            (75, 300, "Advanced"),
            (60, 350, "Intermediate"),
            (35, 450, "Beginner")
        ]
        
        print("\nTest Predictions:")
        print("-" * 40)
        
        for score, time, expected in test_cases:
            predicted = db.predict_skill_level(score, time)
            status = "✅" if predicted.lower() == expected.lower() else "❌"
            print(f"{status} Score: {score:>3}%, Time: {time:>3}s → {predicted.capitalize()}")
        
    else:
        print("❌ Failed to seed performance data")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())