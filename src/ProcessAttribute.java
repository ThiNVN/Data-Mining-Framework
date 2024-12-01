import java.io.File;

import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ProcessAttribute {

    public void csvToArff() throws Exception {
        //Load csv file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/wind_dataset.csv"));
        Instances dataset = loader.getDataSet();

        //Save as arff format
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File("src/output.arff"));
        saver.writeBatch();
    }

    public Instances removeAttribute(Instances data) throws Exception {
        //Remove 1st attribute
        String[] opts = new String[]{"-R","1"};
        Remove remove = new Remove();
        remove.setOptions(opts);
        remove.setInputFormat(data);

        return Filter.useFilter(data, remove);
    }
}
