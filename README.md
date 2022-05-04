<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx">
    <img src="images/logo.png" alt="Logo" width="160" height="160">
  </a>

<h3 align="center">DimRed</h3>

  <p align="center">
    Dimensionality reduction methods 
    <br />
    <a href="https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx">View Demo</a>
    ·
    <a href="https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/issues">Report Bug</a>
    ·
    <a href="https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This is where I describe the dimensionality reduction methods implemented here.
- TSNE
- GraphDR

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Getting started instructions...

### Prerequisites

Requirements for running the modules are listed in `requirements.txt`. These can be directly installed with pip using the following command:
```sh
pip install -r requirements.txt
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx.git
   ```
2. Do some other stuff 
   ```sh
   git good at programing
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

CLI Usage:
  dimred.py tsne <datafilepath> <labelsfilepath>
  dimred.py tsne [options] 
  dimred.py graphdr [options] 

options:
    -p --plot=<bool>   Plot the output          [default: True]
    -s --save=<bool>   Save plot                [default: True]
    --htmlPlot=<bool>  Plot saved as html       [default: True]
    --plot3d=<bool>    Plot in 3D               [default: False]
    --demo=<bool>      Load and run demo        [default: False]

datafilepath:
    --datafilepath=<str>  read in data from file path

labelsfilepath:    
    --labelsfilepath=<str> read in labels from file path

Examples:

```sh
python dimred.py tsne ./data/demo_mnist2500_X.txt ./data/demo_mnist2500_labels.txt
```

TSNE
```sh
# Demo data
python dimred.py tsne --demo=True

# Or specify your own file paths
python dimred.py tsne ./data/demo_mnist2500_X.txt ./data/demo_mnist2500_labels.txt  
```



GraphDR
```sh
# Demo Data:
# 2D Html output
python dimred.py graphdr --demo=True
# 3D Html output
python dimred.py graphdr --demo=True --plot3d=True
```


_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Full CLI implementation 
    - [ ] TSNE attributes
    - [ ] GraphDR
- [ ] 'Demo' version
- [ ] Figure out how to document complicated things
    - [ ] Gains
    - [ ] Beta/Perplexity
    - [ ] Tolerance
- [ ] Clean repo
    - [x] Separate into folders
        - data
        - images
- [ ] Implement GraphDR
- [ ] Implement Plotly
- [ ] Implement Dash interface

See the [open issues](https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the GNU License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Austin Marckx - austinmarckx@gmail.com

Project Link: [https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx](https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Dr. Jian Zhou](jian.zhou@utsouthwestern.edu)
* [Chenlai Shi](chenlai.shi@utsouthwestern.edu)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx.svg?style=for-the-badge
[contributors-url]: https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx.svg?style=for-the-badge
[forks-url]: https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/network/members
[stars-shield]: https://img.shields.io/github/stars/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx.svg?style=for-the-badge
[stars-url]: https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/stargazers
[issues-shield]: https://img.shields.io/github/issues/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx.svg?style=for-the-badge
[issues-url]: https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/issues
[license-shield]: https://img.shields.io/github/license/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx.svg?style=for-the-badge
[license-url]: https://github.com/UTSW-Software-Engineering-Course-2022/module_1_austinmarckx/blob/master/LICENSE.txts
[product-screenshot]: images/screenshot.png