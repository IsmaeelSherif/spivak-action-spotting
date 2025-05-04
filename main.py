from bin.spotting_challenge_commands import \
    commands_spotting_challenge, print_commands, BAIDU_TWO_FEATURE_NAME, \
    BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO


# Commands for running just testing with the models from the zoo:
baidu_two_challenge_commands_zoo = commands_spotting_challenge(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME,
    run_name=RUN_NAME_ZOO, do_train=False)

print_commands(baidu_two_challenge_commands_zoo)