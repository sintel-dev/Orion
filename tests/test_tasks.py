from pytest import fixture

from tasks import _get_minimum_versions, _get_package_names


@fixture
def dependencies():
    return [
        "numpy>=1.20.0,<2;python_version<'3.10'",
        "numpy>=1.23.3,<2;python_version>='3.10'",
        "pandas>=1.2.0,<2;python_version<'3.10'",
        "pandas>=1.3.0,<2;python_version>='3.10'",
        'tensorflow>=2.2,<2.15',
        'pandas @ git+https://github.com/pandas-dev/pandas.git@master#egg=pandas'
    ]


def test_get_package_names(dependencies):
    # Run
    packages = _get_package_names(dependencies)

    # Assert
    expected = [
        'numpy',
        'numpy',
        'pandas',
        'pandas',
        'tensorflow',
        'pandas'
    ]

    assert packages == expected


def test_get_minimum_versions(dependencies):
    # Run
    minimum_versions_39 = _get_minimum_versions(dependencies, '3.9')
    minimum_versions_310 = _get_minimum_versions(dependencies, '3.10')

    # Assert
    expected_versions_39 = [
        'numpy==1.20.0',
        'pandas @ git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'tensorflow==2.2',
    ]
    expected_versions_310 = [
        'numpy==1.23.3',
        'pandas @ git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'tensorflow==2.2',
    ]

    assert minimum_versions_39 == expected_versions_39
    assert minimum_versions_310 == expected_versions_310
