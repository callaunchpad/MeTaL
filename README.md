# MeTaL
#### Launchpad Fall 2018
## Virtual Environments and Dependencies
We strongly encourage building dependencies within a virtual environment. To use the Python virtual environments, follow [this guide on python environments](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/), or to use Conda environments, follow [this guide on conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html).

Once inside your environment, from the base directory of this repo, run

```
pip install -r requirements.txt
```
This should add any required dependencies to your virtual environment.

## Contributing Code and Workflow
**Please make sure all commits have meaningful comments!**

Do not use `git add .` to add files. This adds files that should potentially be untracked. Instead, use
```
git commit -am "commit message"
```
This adds and commits all modified files that are tracked. If you need to track new files, then use
```
git add new_file_name_here
```

When contributing code, make sure you code is documented. Each file should have a header explaining the contents within a few sentences, as well as the names of all contributing authors. Each class, function and method should be documented with the following style:

```
class bar():
  """ One line description on bar class"""
  def foo(self, arg1, arg2):
    """
    One line description of the functionality of foo.
    
    Args:
      arg1: Type and contents of arg1
      arg2: Type and contents of arg2
    Returns:
      Type and contents of what is returned
    @Authors: Names of authors, comma seperated
    """
    ...
```
Note that if the function is part of a class, the `self` argument does not need to be documented.

If you are contributing to a file for the first time, please be sure to add your name to all locations where you contributed.

Additionally, please restrict your lines to 80 characters in length. to continue to a new line, use parentheses. The following lines should start the column after the openning parenthese. Here is an example (also an example of KL Divergence for a variational autoencoder).

```
latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                   - tf.square(self.z_mean) 
                                   - tf.exp(self.z_log_sigma_sq), 1)
```


If you are not a member of the MeTaL team, feel free to submit pull requests and/or contact members of the MeTaL team.

## Code Structure
### Algorithms:
This contains any code pertaining to learning/neural network algorithms and architectures.
