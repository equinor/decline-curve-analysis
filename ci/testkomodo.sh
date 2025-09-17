copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    cp $CI_SOURCE_ROOT/pyproject.toml $CI_TEST_ROOT
}

install_test_dependencies () {
    pip install ".[test]"
}

start_tests () {
    pytest -n auto --ert-integration
}