import os
import sys
# Get the directory containing the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (spivak-action-spotting)
parent_dir = os.path.dirname(current_dir) + '/spivak-action-spotting'
# Add the parent directory to sys.path
sys.path.append(parent_dir)
print('added to sys.path', parent_dir)


from bin.spotting_challenge_commands import \
    commands_spotting_challenge, print_commands, BAIDU_TWO_FEATURE_NAME, \
    BAIDU_TWO_FEATURES_DIR, RUN_NAME_ZOO


# Commands for running just testing with the models from the zoo:
baidu_two_challenge_commands_zoo = commands_spotting_challenge(
    BAIDU_TWO_FEATURES_DIR, BAIDU_TWO_FEATURE_NAME,
    run_name=RUN_NAME_ZOO, do_train=False)

print_commands(baidu_two_challenge_commands_zoo)