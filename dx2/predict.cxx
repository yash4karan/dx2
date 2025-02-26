#include <dx2/predict/predict.hpp>
#include <iostream>

int main(int argc, char **argv) {
    // Initialize an ArgumentParser object: 
    /*
    parser = ArgumentParser(
                usage=usage,
                phil=phil_scope,
                epilog=help_message,
                check_format=False,
                read_experiments=True,
                )
    */

    // Extract params, options form this parser
    /*
    params, options = parser.parse_args(args, show_diff_phil=True)
    */

    // Flatten experiments into a 1D array
    /*
    experiments = flatten_experiments(params.input.experiments)
    */

    // If length of experiments in 0, print a help message and exit
    /*
    if len(experiments) == 0:
        parser.print_help()
        return
    */

    // Create an empty reflection table to store predictions in
    /*
    predicted_all = reflection_table()
    */

    // Populate `predicated_all` with predictions
    /*
    for i_expt, expt in enumerate(experiments):
    */

        // Predict reflections outside the range of the scan if buffer_size > 0
        /*
        if params.buffer_size > 0:
            expt.scan.set_image_range(
                (
                    expt.scan.get_image_range()[0] - params.buffer_size,
                    expt.scan.get_image_range()[1] + params.buffer_size,
                )
            )
            expt.scan.set_oscillation(
                (
                    expt.scan.get_oscillation()[0] - params.buffer_size * oscillation[1],
                    expt.scan.get_oscillation()[1],
                )
            )
        */

        // Create reflection table using `expt`, `params.force_static`, `params.d_min`
        /*
        predicted = flex.reflection_table.from_predictions(
            expt, force_static=params.force_static, dmin=params.d_min
        )
        */

        // Copy experiment identifiers verbatim
        /*
        predicted.experiment_identifiers()[i_expt] = experiments[i_expt].identifier
        predicted["id"] = flex.int(len(predicted), i_expt)
        predicted_all.extend(predicted)
        */
    
    // If not ignoring shadows, look for reflections in the masked region
    /*
    if not params.ignore_shadows:
        try:
            experiments = ExperimentListFactory.from_json(
                experiments.as_json(), check_format=True
            )
        except OSError as e:
            sys.exit(
                f"Unable to read image data. Please check {e.filename} is accessible"
            )
        shadowed = filter_shadowed_reflections(
            experiments, predicted_all, experiment_goniometer=True
        )
        predicted_all = predicted_all.select(~shadowed)
    */

    // Try to find bounding boxed for each experiment
    /*
    try:
        predicted_all.compute_bbox(experiments)
    except Exception:
        pass
    */

    // Save reflections to file
    /*
    Command.start(f"Saving {len(predicted_all)} reflections to {params.output}")
    predicted_all.as_file(params.output)
    Command.end(f"Saved {len(predicted_all)} reflections to {params.output}")
    */

    return 0;
}