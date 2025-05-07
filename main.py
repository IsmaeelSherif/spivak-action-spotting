
from bin.spotting_challenge_commands import \
    commands_spotting_challenge, commands_spotting_challenge_validated, RESNET_NORMALIZED_FEATURE_NAME, RESNET_NORMALIZED_FEATURES_DIR, print_commands, BAIDU_TWO_FEATURE_NAME, \
    BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO


# # Commands for running just testing with the models from the zoo:
# baidu_two_challenge_commands_zoo = commands_spotting_challenge(
#     BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME,
#     run_name=RUN_NAME_ZOO, do_train=False)

# print_commands(baidu_two_challenge_commands_zoo)


# Commands for running just testing with the models from the zoo:
# resnet_normalized_challenge_validated_commands_zoo = \
#     commands_spotting_challenge_validated(
#         RESNET_NORMALIZED_FEATURES_DIR, RESNET_NORMALIZED_FEATURE_NAME,
#         run_name=RUN_NAME_ZOO, do_train=False)
# print_commands(resnet_normalized_challenge_validated_commands_zoo)


from bin.spotting_challenge_commands import \
    command_resample_baidu, commands_normalize_resnet, print_commands

resample_command = command_resample_baidu()
print(resample_command)
# You can run the command with:
# resample_command.run()
normalize_commands = commands_normalize_resnet()
print_commands(normalize_commands)