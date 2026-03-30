from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    settings.prj_dir = '/content/SDTrack/SDTrack-Event'
    settings.network_path = '/content/SDTrack/SDTrack-Event/pretrained_models'
    settings.result_plot_path = '/content/SDTrack/SDTrack-Event/output/test/result_plots'
    settings.results_path = '/content/SDTrack/SDTrack-Event/output/test/tracking_results'
    settings.save_dir = '/content/SDTrack/SDTrack-Event/output'
    settings.segmentation_path = '/content/SDTrack/SDTrack-Event/output/test/segmentation_results'
    settings.eotb_path = '/content/SDTrack/SDTrack-Event/data/test'

    return settings
