import java.io.File;

import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class ProcessDataset {

    public static void csvToArff(String filePath) throws Exception {
        //Load csv file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances dataset = loader.getDataSet();

        //Save as arff format
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("src/data/weather_classification.arff"));
        saver.writeBatch();
    }

    public static void saveInstances(Instances dataset, String filePath) throws Exception {
        //Save as arff format
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File(filePath));
        saver.writeBatch();
    }

    public static Instances underSampling(Instances dataset) throws Exception {
        Resample resample = new Resample();
        resample.setInputFormat(dataset);
        resample.setBiasToUniformClass(1.0);
        resample.setSampleSizePercent(100);
        Instances underSampledDataset = Filter.useFilter(dataset, resample);
        return underSampledDataset;
    }

    public static Instances removeAttribute(Instances data, String removeAttributes) throws Exception {
        //Remove 1st attribute
        String[] opts = new String[]{"-R", removeAttributes};
        Remove remove = new Remove();
        remove.setOptions(opts);
        remove.setInputFormat(data);

        return Filter.useFilter(data, remove);
    }

}
