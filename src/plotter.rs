use plotters::prelude::*;

pub fn plot_epoch_accuracy(epochs: &[usize], accuracies: &[f32], output_file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(output_file_path, (640, 480))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Epoch vs. Accuracy", ("sans-serif", 40))
        .margin(5)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .build_cartesian_2d(
            *epochs.first().unwrap() as f32..*epochs.last().unwrap() as f32,
            0f32..1f32,
        )?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy")
        .draw()?;

    chart.draw_series(LineSeries::new(
        epochs.iter().map(|&x| x as f32).zip(accuracies.iter().copied()),
        &RED,
    ))?.label("Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root_area.present()?;
    Ok(())
}
