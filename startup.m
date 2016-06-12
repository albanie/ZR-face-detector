% Add paths to configuration script
addpath('~/coding/src/corelib/utils');

paths = loadPaths();
addpath(genpath(paths.CORE));
run(fullfile(paths.VLFEAT, 'toolbox', 'vl_setup'));
