from setuptools import setup, find_packages

from setuptools_scm import get_version

if __name__ == "__main__":
    lpackages = find_packages('.')
    print(lpackages)
    
    print("\nINFO: get_version:")
    try:
        # local_scheme for pypi:
        print(get_version(root='.', relative_to=__file__,
                          local_scheme="no-local-version"))
    except:
        
        raise(BaseException("setuptools_scm get_version bug,"
                            + " update to v 3.5.0 needed:"
                            + " pip uninstall setuptools_scm"
                            + "\n pip install -Iv setuptools_scm>=3.5.0"))

    # for description:
    with open("README.md") as f:
        long_description = f.read()

    # for requirements:
    with open("requirements.txt") as f:
        s_reguirements = f.read()
    requirements = s_reguirements.split("\n")
    # requirements = [';'.join([r for r in requirements if len(r) > 0])]
    requirements = [r for r in requirements if len(r) > 0]
    print("\nINFO: requirements:")
    print(requirements)

    installed = setup(
        
        name="tratools",

        # use version (only) from setuptools_scm:
        use_scm_version={
            "root": ".",
            "relative_to": __file__,
            "local_scheme": "no-local-version"
        },

        # use_scm_version=True,
        # version=get_version(root='.', relative_to=__file__),

        author="tra-group",
        author_email="",
        description="some helpful tools",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/tokentranslator-group/tratools",
        packages=lpackages,
        include_package_data=True,
        
        # exclude_package_data={"hybriddomain.tests":
        #                       ["*.json", "*.ipynb",
        #                        "problems/*", "settings/*"]},
        # for ``include_package_data`` to work accordingly to git:
        # setuptools_scm will register itself as setuptools plug in,
        # with use of ``entry_points``, so it's git/hg ``file_finders``
        # will be implicitly used for finding only git/hg managable files
        # (that not in .{git/hg}ignore) when ``include_package_data``
        # or ``package_data`` attributes of setuptools.setup() was used.
        # (see https://github.com/pypa/setuptools_scm/blob/master/setup.py)

        # this will automatically used due to setup_requires setuptools_scm
        # (it will add it's entry_points to the setuptools during install):
        # entry_points="""
        # [setuptools.file_finders]
        # setuptools_scm = setuptools_scm.integration:find_files
        # """,

        setup_requires=['setuptools_scm >= 3.5.0'],
        # setup_requires=[ "setuptools_git >= 0.3", ],
        install_requires=requirements
        
    )
    
