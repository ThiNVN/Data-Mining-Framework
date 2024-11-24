import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class Cleaner {

    public static void main(String[] args) {
        String inputFilePath = "data/cleaned_weather_data.csv";
        String cleanedFilePath = "src/data/cleaned_weather_data.csv"; // File to save the cleaned data
        List<String> columnsToMine = Arrays.asList("Weather Type", "Cloud Cover", "Season"); // Specify categorical
        // columns
        int minSupport = 2; // Minimum support threshold for frequent sequences
        double trainTestRatio = 0.8; // 80% training, 20% testing

        // Step 1: Clean the Data
        cleanData(inputFilePath, cleanedFilePath);

        // Step 2: Parse Sequences
        List<List<String>> sequences = parseSequences(cleanedFilePath, columnsToMine);
        if (sequences.isEmpty()) {
            System.out.println("No valid sequences found. Please check the input file.");
            return;
        }

        // Step 3: Split Data
        Map<String, List<List<String>>> splitData = splitData(sequences, trainTestRatio);
        List<List<String>> trainingData = splitData.get("train");
        List<List<String>> testData = splitData.get("test");

        System.out.println("Training Data Size: " + trainingData.size());
        System.out.println("Test Data Size: " + testData.size());

        // Step 4: Calculate Statistics
        calculateStatistics(cleanedFilePath);

        // Step 5: Mine Frequent Subsequences
        Map<List<String>, Integer> frequentSequences = mineSequences(trainingData, minSupport);
        System.out.println("Frequent Subsequences:");
        for (Map.Entry<List<String>, Integer> entry : frequentSequences.entrySet()) {
            System.out.println("Sequence: " + entry.getKey() + ", Support: " + entry.getValue());
        }
    }

    public static void cleanData(String inputFilePath, String outputFilePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {

            String header = reader.readLine();
            writer.write(header + "\n");

            String[] headers = header.split(",");
            List<List<Double>> numericColumns = new ArrayList<>();
            for (int i = 0; i < headers.length; i++) {
                numericColumns.add(new ArrayList<>());
            }

            List<String[]> rows = new ArrayList<>();
            String line;

            // Read and identify numeric values
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                rows.add(fields);
                for (int i = 0; i < fields.length; i++) {
                    try {
                        double value = Double.parseDouble(fields[i].trim());
                        numericColumns.get(i).add(value);
                    } catch (NumberFormatException ignored) {
                    }
                }
            }

            // Calculate column means
            double[] columnMeans = new double[headers.length];
            for (int i = 0; i < headers.length; i++) {
                List<Double> column = numericColumns.get(i);
                columnMeans[i] = column.isEmpty() ? 0.0
                        : column.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            }

            // Replace missing values and write cleaned data
            for (String[] fields : rows) {
                for (int i = 0; i < fields.length; i++) {
                    if (fields[i].trim().isEmpty() || fields[i].trim().equalsIgnoreCase("null")
                            || fields[i].trim().equalsIgnoreCase("undefined")) {
                        if (!Double.isNaN(columnMeans[i]) && columnMeans[i] != 0.0) {
                            fields[i] = String.valueOf(columnMeans[i]);
                        }
                    }
                }
                writer.write(String.join(",", fields) + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Parse categorical sequences from the CSV file
    public static List<List<String>> parseSequences(String inputFilePath, List<String> columnsToMine) {
        List<List<String>> sequences = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath))) {
            String line = reader.readLine(); // Read header
            String[] headers = line.split(",");

            // Determine indices of columns to mine
            List<Integer> columnIndices = new ArrayList<>();
            for (String column : columnsToMine) {
                for (int i = 0; i < headers.length; i++) {
                    if (headers[i].trim().equalsIgnoreCase(column)) {
                        columnIndices.add(i);
                        break;
                    }
                }
            }

            // Read and parse sequences
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                List<String> sequence = new ArrayList<>();
                for (int index : columnIndices) {
                    sequence.add(fields[index].trim());
                }
                sequences.add(sequence);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sequences;
    }

    // Split data into training and test sets
    public static Map<String, List<List<String>>> splitData(List<List<String>> sequences, double trainTestRatio) {
        Map<String, List<List<String>>> splitData = new HashMap<>();
        Collections.shuffle(sequences, new Random()); // Shuffle sequences randomly

        int trainSize = (int) (sequences.size() * trainTestRatio);
        List<List<String>> trainingData = sequences.subList(0, trainSize);
        List<List<String>> testData = sequences.subList(trainSize, sequences.size());

        splitData.put("train", new ArrayList<>(trainingData));
        splitData.put("test", new ArrayList<>(testData));
        return splitData;
    }

    // Calculate statistics for numeric columns in the CSV file
    public static void calculateStatistics(String inputFilePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath))) {
            String line = reader.readLine(); // Read header
            String[] headers = line.split(",");

            List<List<Double>> numericColumns = new ArrayList<>();
            for (int i = 0; i < headers.length; i++) {
                numericColumns.add(new ArrayList<>());
            }

            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                for (int i = 0; i < fields.length; i++) {
                    try {
                        double value = Double.parseDouble(fields[i].trim());
                        numericColumns.get(i).add(value);
                    } catch (NumberFormatException ignored) {
                    }
                }
            }

            System.out.println("Basic Statistics:");
            for (int i = 0; i < headers.length; i++) {
                List<Double> column = numericColumns.get(i);
                if (!column.isEmpty()) {
                    double mean = column.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                    double median = calculateMedian(column);
                    double min = Collections.min(column);
                    double max = Collections.max(column);
                    double stdDev = calculateStandardDeviation(column, mean);

                    System.out.printf("%s -> Mean: %.2f, Median: %.2f, Min: %.2f, Max: %.2f, StdDev: %.2f%n",
                            headers[i], mean, median, min, max, stdDev);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double calculateMedian(List<Double> values) {
        Collections.sort(values);
        int size = values.size();
        if (size % 2 == 0) {
            return (values.get(size / 2 - 1) + values.get(size / 2)) / 2.0;
        } else {
            return values.get(size / 2);
        }
    }

    private static double calculateStandardDeviation(List<Double> values, double mean) {
        double variance = values.stream().mapToDouble(v -> Math.pow(v - mean, 2)).sum() / values.size();
        return Math.sqrt(variance);
    }

    // Mine frequent sequences using an Apriori-like approach
    public static Map<List<String>, Integer> mineSequences(List<List<String>> sequences, int minSupport) {
        Map<List<String>, Integer> allFrequentSequences = new HashMap<>();

        // Generate initial 1-item sequences and count their occurrences
        Map<List<String>, Integer> candidateSequences = generateInitialCandidates(sequences);

        while (!candidateSequences.isEmpty()) {
            // Filter candidates based on minimum support
            Map<List<String>, Integer> frequentSequences = new HashMap<>();
            for (Map.Entry<List<String>, Integer> entry : candidateSequences.entrySet()) {
                if (entry.getValue() >= minSupport) {
                    frequentSequences.put(entry.getKey(), entry.getValue());
                }
            }

            // Add to global result
            allFrequentSequences.putAll(frequentSequences);

            // Generate next-level candidates
            candidateSequences = generateNextCandidates(frequentSequences.keySet());
            countCandidateOccurrences(candidateSequences, sequences);
        }

        return allFrequentSequences;
    }

    private static Map<List<String>, Integer> generateInitialCandidates(List<List<String>> sequences) {
        Map<List<String>, Integer> candidates = new HashMap<>();
        for (List<String> sequence : sequences) {
            for (String item : sequence) {
                List<String> singleItemSequence = Collections.singletonList(item);
                candidates.put(singleItemSequence, candidates.getOrDefault(singleItemSequence, 0) + 1);
            }
        }
        return candidates;
    }

    private static Map<List<String>, Integer> generateNextCandidates(Set<List<String>> frequentSequences) {
        Map<List<String>, Integer> candidates = new HashMap<>();
        List<List<String>> frequentList = new ArrayList<>(frequentSequences);

        for (int i = 0; i < frequentList.size(); i++) {
            for (int j = i + 1; j < frequentList.size(); j++) {
                List<String> candidate = new ArrayList<>(frequentList.get(i));
                candidate.addAll(frequentList.get(j));
                Set<String> uniqueItems = new LinkedHashSet<>(candidate);
                candidates.put(new ArrayList<>(uniqueItems), 0);
            }
        }
        return candidates;
    }

    private static void countCandidateOccurrences(Map<List<String>, Integer> candidates, List<List<String>> sequences) {
        for (List<String> sequence : sequences) {
            for (Map.Entry<List<String>, Integer> entry : candidates.entrySet()) {
                if (isSubsequence(entry.getKey(), sequence)) {
                    candidates.put(entry.getKey(), entry.getValue() + 1);
                }
            }
        }
    }

    private static boolean isSubsequence(List<String> candidate, List<String> sequence) {
        int index = 0;
        for (String item : sequence) {
            if (index < candidate.size() && item.equals(candidate.get(index))) {
                index++;
            }
        }
        return index == candidate.size();
    }
}
