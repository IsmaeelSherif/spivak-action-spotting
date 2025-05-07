
from bin.spotting_challenge_commands import \
    commands_spotting_challenge, commands_spotting_challenge_validated, RESNET_NORMALIZED_FEATURE_NAME, RESNET_NORMALIZED_FEATURES_DIR, print_commands, BAIDU_TWO_FEATURE_NAME, \
    BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO


# Commands for running just testing with the models from the zoo:
baidu_two_challenge_commands_zoo = commands_spotting_challenge(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME,
    run_name=RUN_NAME_ZOO, do_train=False)

print_commands(baidu_two_challenge_commands_zoo)


# Commands for running just testing with the models from the zoo:
resnet_normalized_challenge_validated_commands_zoo = \
    commands_spotting_challenge_validated(
        RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME,
        run_name=RUN_NAME_ZOO, do_train=False)
print_commands(resnet_normalized_challenge_validated_commands_zoo)