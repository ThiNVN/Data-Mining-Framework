
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class Cleaner {

	public static void main(String[] args) {
		String inputFilePath = "weather_classification_data.csv";
		String cleanedFilePath = "cleaned_weather_data.csv";
		String trainingFilePath = "training_set.csv";
		String testFilePath = "test_set.csv";
		double trainingRatio = 0.8;

		try {
			// Step 1: Clean Data and Calculate Modes
			Map<String, String> columnModes = cleanDataAndCalculateModes(inputFilePath, cleanedFilePath);

			// Step 2: Print Modes
			System.out.println("\nModes of Columns:");
			columnModes.forEach((column, mode) -> System.out.printf("%s -> Mode: %s%n", column, mode));

			// Step 3: Load cleaned data
			List<String[]> cleanedData = readData(cleanedFilePath);

			// Extract headers from cleaned data
			String[] headers = cleanedData.get(0);
			cleanedData.remove(0); // Remove header row for processing

			// Normalize data
			List<String[]> normalizedData = normalizeData(cleanedData, headers);

			// Encode categorical data
			List<String[]> encodedData = encodeCategoricalData(normalizedData, headers);

			// Feature selection
			List<String> selectedFeatures = Arrays.asList("Temperature", "Humidity");
			List<String[]> finalData = selectFeatures(encodedData, headers, selectedFeatures);

			// Step 4: Identify Patterns
			identifyPatterns(cleanedFilePath);

			// Step 5: Split data into training and test sets
			splitData(finalData, headers, trainingFilePath, testFilePath, trainingRatio);

		} catch (Exception e) {
			System.err.println("An error occurred: " + e.getMessage());
			e.printStackTrace();
		}
	}

	public static Map<String, String> cleanDataAndCalculateModes(String inputFilePath, String outputFilePath)
			throws IOException {
		Map<String, String> columnModes = new HashMap<>();
		Set<String> uniqueRows = new HashSet<>();
		List<Map<String, Integer>> valueCounts = new ArrayList<>();
		List<String[]> rows = new ArrayList<>();
		String[] headers;

		try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
				BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {

			// Read header
			String headerLine = reader.readLine();
			if (headerLine == null)
				throw new IOException("Input file is empty.");
			headers = headerLine.split(",");
			writer.write(headerLine + "\n");

			for (String header : headers) {
				valueCounts.add(new HashMap<>());
			}

			// Process rows
			String line;
			while ((line = reader.readLine()) != null) {
				if (uniqueRows.add(line)) {
					String[] fields = line.split(",");
					rows.add(fields);

					for (int i = 0; i < fields.length; i++) {
						String value = fields[i].trim();
						if (!isMissingOrInvalid(value)) {
							valueCounts.get(i).merge(value, 1, Integer::sum);
						}
					}
				}
			}

			// Calculate modes
			for (int i = 0; i < headers.length; i++) {
				Map<String, Integer> counts = valueCounts.get(i);
				columnModes.put(headers[i], counts.entrySet().stream().max(Map.Entry.comparingByValue())
						.map(Map.Entry::getKey).orElse("UNKNOWN"));
			}

			// Fill missing values and write cleaned data
			for (String[] fields : rows) {
				for (int i = 0; i < fields.length; i++) {
					if (isMissingOrInvalid(fields[i])) {
						fields[i] = columnModes.get(headers[i]);
					}
				}
				writer.write(String.join(",", fields) + "\n");
			}

			System.out.println("Data cleaned and written to: " + outputFilePath);
		}

		return columnModes;
	}

	public static boolean isMissingOrInvalid(String value) {
		return value == null || value.trim().isEmpty() || value.equalsIgnoreCase("null")
				|| value.equalsIgnoreCase("undefined");
	}

	public static List<String[]> readData(String filePath) throws IOException {
		List<String[]> data = new ArrayList<>();
		try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
			String line;
			while ((line = reader.readLine()) != null) {
				data.add(line.split(","));
			}
		}
		return data;
	}

	public static List<String[]> normalizeData(List<String[]> data, String[] headers) {
		int numColumns = headers.length;
		double[] minValues = new double[numColumns];
		double[] maxValues = new double[numColumns];
		Arrays.fill(minValues, Double.MAX_VALUE);
		Arrays.fill(maxValues, Double.MIN_VALUE);

		// Find min and max values
		for (String[] row : data) {
			for (int i = 0; i < row.length; i++) {
				try {
					double value = Double.parseDouble(row[i]);
					minValues[i] = Math.min(minValues[i], value);
					maxValues[i] = Math.max(maxValues[i], value);
				} catch (NumberFormatException ignored) {
				}
			}
		}

		// Normalize data
		List<String[]> normalizedData = new ArrayList<>();
		for (String[] row : data) {
			String[] normalizedRow = new String[row.length];
			for (int i = 0; i < row.length; i++) {
				try {
					double value = Double.parseDouble(row[i]);
					if (maxValues[i] != minValues[i]) {
						normalizedRow[i] = String.valueOf((value - minValues[i]) / (maxValues[i] - minValues[i]));
					} else {
						normalizedRow[i] = "0"; // Default for zero range
					}
				} catch (NumberFormatException e) {
					normalizedRow[i] = row[i]; // Retain categorical data
				}
			}
			normalizedData.add(normalizedRow);
		}

		return normalizedData;
	}

	public static List<String[]> encodeCategoricalData(List<String[]> data, String[] headers) {
		Map<String, Map<String, Integer>> encoders = new HashMap<>();
		for (int i = 0; i < headers.length; i++) {
			Map<String, Integer> encoder = new HashMap<>();
			int index = 0;
			for (String[] row : data) {
				String value = row[i];
				if (!encoder.containsKey(value)) {
					encoder.put(value, index++);
				}
			}
			encoders.put(headers[i], encoder);
		}

		List<String[]> encodedData = new ArrayList<>();
		for (String[] row : data) {
			String[] encodedRow = new String[row.length];
			for (int i = 0; i < row.length; i++) {
				Map<String, Integer> encoder = encoders.get(headers[i]);
				encodedRow[i] = String.valueOf(encoder.getOrDefault(row[i], -1)); // -1 for unknown values
			}
			encodedData.add(encodedRow);
		}

		return encodedData;
	}

	public static List<String[]> selectFeatures(List<String[]> data, String[] headers, List<String> selectedFeatures) {
		List<Integer> selectedIndices = new ArrayList<>();
		for (int i = 0; i < headers.length; i++) {
			if (selectedFeatures.contains(headers[i])) {
				selectedIndices.add(i);
			}
		}

		if (selectedIndices.isEmpty()) {
			throw new IllegalArgumentException("No matching features found for selection.");
		}

		List<String[]> reducedData = new ArrayList<>();
		for (String[] row : data) {
			String[] reducedRow = new String[selectedIndices.size()];
			for (int j = 0; j < selectedIndices.size(); j++) {
				reducedRow[j] = row[selectedIndices.get(j)];
			}
			reducedData.add(reducedRow);
		}

		return reducedData;
	}

	public static void identifyPatterns(String inputFilePath) throws IOException {
		try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath))) {
			String[] headers = reader.readLine().split(",");

			List<List<Double>> numericData = new ArrayList<>(headers.length);
			for (int i = 0; i < headers.length; i++) {
				numericData.add(new ArrayList<>());
			}

			String line;
			while ((line = reader.readLine()) != null) {
				String[] fields = line.split(",");
				for (int i = 0; i < fields.length; i++) {
					try {
						numericData.get(i).add(Double.parseDouble(fields[i]));
					} catch (NumberFormatException ignored) {
					}
				}
			}

			System.out.println("\nStatistical Summaries:");
			for (int i = 0; i < headers.length; i++) {
				if (!numericData.get(i).isEmpty()) {
					double mean = numericData.get(i).stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
					double max = numericData.get(i).stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
					double min = numericData.get(i).stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
					System.out.printf("%s: Mean=%.2f, Max=%.2f, Min=%.2f%n", headers[i], mean, max, min);
				}
			}
		}
	}

	public static void splitData(List<String[]> data, String[] headers, String trainingFilePath, String testFilePath,
			double trainingRatio) throws IOException {
		Collections.shuffle(data, new Random(42)); // Fixed seed for reproducibility
		int splitIndex = (int) (data.size() * trainingRatio);

		try (BufferedWriter trainingWriter = new BufferedWriter(new FileWriter(trainingFilePath));
				BufferedWriter testWriter = new BufferedWriter(new FileWriter(testFilePath))) {

			trainingWriter.write(String.join(",", headers) + "\n");
			testWriter.write(String.join(",", headers) + "\n");

			for (int i = 0; i < data.size(); i++) {
				String line = String.join(",", data.get(i)) + "\n";
				if (i < splitIndex) {
					trainingWriter.write(line);
				} else {
					testWriter.write(line);
				}
			}
		}

		System.out.printf("Data split into %s (training) and %s (test).%n", trainingFilePath, testFilePath);
	}
}
