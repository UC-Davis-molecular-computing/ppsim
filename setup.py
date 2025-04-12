from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

# this is ugly, but appears to be standard practice:
# https://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package/17626524#17626524
def extract_version(filename: str):
    with open(filename) as f:
        lines = f.readlines()
    version_comment = '# version line; WARNING: do not remove or change this line or comment'
    for line in lines:
        if version_comment in line:
            idx = line.index(version_comment)
            line_prefix = line[:idx]
            parts = line_prefix.split('=')
            parts = [part.strip() for part in parts]
            version_str = parts[-1]
            version_str = version_str.replace('"', '')
            version_str = version_str.replace("'", '')
            version_str = version_str.strip()
            return version_str
    raise AssertionError(f'could not find version in {filename}')

version = extract_version('ppsim/__version__.py')
print(f'ppsim version = {version}')

setup(
    name="ppsim",
    packages=['ppsim'],
    version=version,
    author="Eric Severson and David Doty",
    description="A package for simulating population protocols.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    url="https://github.com/UC-Davis-molecular-computing/ppsim",
    install_requires=install_requires
)
