use snakers::{
    Tensor,
    Layer,
    Dense,
    ReLU,
    CategoricalCrossEntropy,
    SGD,
    Sequential,
};
use csv;
use std::error::Error;
use std::path::Path;

fn load_mnist(file_path: &Path) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let num_classes = 10;

    for result in reader.records() {
        let record = result?;
        
        let label = record[0].parse::<usize>()?;
        
        let image_features: Vec<f32> = record.iter()
            .skip(1)
            .map(|pixel| pixel.parse::<f32>().unwrap_or(0.0) / 255.0)
            .collect();
        x_data.extend(image_features);
        
        let mut one_hot_label = vec![0.0; num_classes];
        one_hot_label[label] = 1.0;
        y_data.extend(one_hot_label);
    }
    
    let num_samples = y_data.len() / num_classes;
    let num_features = x_data.len() / num_samples;

    let x_tensor = Tensor::from_vec(x_data, vec![num_samples, num_features]);
    let y_tensor = Tensor::from_vec(y_data, vec![num_samples, num_classes]);

    Ok((x_tensor, y_tensor))
}


fn main() -> Result<(), Box<dyn Error>> {
    println!("Loading MNIST data...");
    let (x_train, y_train) = load_mnist(Path::new("./input/mnist_train.csv"))?;
    let (x_test, y_test) = load_mnist(Path::new("./input/mnist_test.csv"))?;
    println!("Data loaded successfully.");
    println!("Training samples: {}, Test samples: {}", x_train.shape[0], x_test.shape[0]);

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Dense::new(784, 128)),
        Box::new(ReLU::new()),
        Box::new(Dense::new(128, 10))
    ];

    let loss = Box::new(CategoricalCrossEntropy::new());
    let optimizer = Box::new(SGD::new(0.01));

    let mut model = Sequential::new(layers, loss, optimizer);

    println!("\nStarting training...");
    model.fit(&x_train, &y_train, 5, 32);
    println!("Training finished.");

    println!("\nEvaluating model on test data...");
    let predictions = model.predict(&x_test);
    
    let mut correct_predictions = 0;
    let y_pred_data = predictions.read();
    let y_true_data = y_test.read();
    let num_classes = y_test.shape[1];

    for i in 0..y_test.shape[0] {
        let pred_row_start = i * num_classes;
        let true_row_start = i * num_classes;

        let pred_slice = &y_pred_data[pred_row_start..pred_row_start + num_classes];
        let true_slice = &y_true_data[true_row_start..true_row_start + num_classes];

        let pred_label = pred_slice.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        let true_label = true_slice.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        
        if pred_label == true_label {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / y_test.shape[0] as f32;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    
    Ok(())
}
