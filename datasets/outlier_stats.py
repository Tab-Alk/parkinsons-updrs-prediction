# Add this as a new cell after Cell 23 to get outlier statistics

# Calculate outlier counts using IQR method
print("OUTLIER STATISTICS (IQR Method - 1.5×IQR)")
print("=" * 70)
print(f"{'Feature':<20} {'Outliers':<10} {'% of Data':<10} {'Total Samples':<15}")
print("-" * 70)

outlier_summary = []

for feature in voice_features:
    Q1 = parkinsons[feature].quantile(0.25)
    Q3 = parkinsons[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = parkinsons[(parkinsons[feature] < lower_bound) | (parkinsons[feature] > upper_bound)]
    count = len(outliers)
    pct = (count / len(parkinsons)) * 100

    print(f"{feature:<20} {count:<10} {pct:<10.1f} {len(parkinsons):<15}")

    outlier_summary.append({
        'feature': feature,
        'count': count,
        'pct': pct
    })

# Summary
total_outliers = sum([x['count'] for x in outlier_summary])
print("=" * 70)
print(f"\nTotal outlier data points across all features: {total_outliers}")
print(f"Dataset has {len(parkinsons)} samples")
print(f"\nFeatures with most outliers:")
sorted_summary = sorted(outlier_summary, key=lambda x: x['pct'], reverse=True)
for item in sorted_summary[:5]:
    print(f"  {item['feature']}: {item['count']} ({item['pct']:.1f}%)")
