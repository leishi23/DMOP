
<h1 align="center">
  <br>
  <a href="https://www.tencent.com/zh-cn/business/robotics.html"><img src="https://avatars.githubusercontent.com/u/87681266?s=280&v=4" alt="Dexterous Manipulation Online Planning Repo" width="300", height="200"></a>
  <br>
  Dexterous Manipulation Online Planning Repo
  <br>
</h1>

<!-- <h4 align="center">A minimal Markdown Editor desktop app built on top of <a href="http://electron.atom.io" target="_blank">Electron</a>.</h4> -->

<p align="center">
  <a href="https://github.com/leishi23"><img src="https://badges.gitter.im/amitmerchant1990/electron-markdownify.svg"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#Contents">Contents</a> 
</p>

![screenshot](https://media1.giphy.com/media/3mfxH0nbfVFLt1gTpq/giphy.gif)

## Key Features

* Comprehensive and robust manipulation of objects
* Fast enough to support online computation and replanning

## Contents

- `adaptation.py`: to calculate the final time `t_f`, process force and torque `f*` and `tau*` to be applied to the object.
  - Input: initial velocity, initial angular velocity, mass, inertia, max force vector, max torque vector, final velocity, final angular velocity
  - Output: final time, list of force, list of torque
- `forward_dynamics.py`: to calculate the final state (**_position/velocity/orientation/angular-velocity_**) and state of each step of the object.
  - Input: initial state, mass, inertia, maximum force, maximum torque, ideal final velocity, ideal final angular velocity, number of steps, delta,  gravity
  - Output: final time, final state([position/orientation/velocity/angular velocity]), state at each time step
- `trans.py`: to transform the state from object center to contact point.
  - Input: center state, contact point
  - Output: contact state
- `impact.py`: to get the object state after impact.
    - Input: state before impact, enabled state when impact
    - Output: object state after impact
- `parabola.py`: to get the object state after parabola.
    - Input: state before parabola, final time 
    - Output: object state after parabola
- `approach_pos.py`: to calculate the process position of robot and object. Guide the robot with object.
  - Input: weight of robot, weight of object, robot initial position, robot initial velocity, object initial position, object initial velocity, object initial acceleration, final time 
  - Output: process position of robot and object
- `approach_alt.py`: to calculate the process altitude of robot and object. Guide the robot with object.
  - Input: object's initial position/orientation/angular velocity, any given time tf_K, mass, inertia, torque_max, force_max, iterate time
  - Output: object and robot orientation list of each step

![image](https://github.com/leishi23/Dexterous-Manipulation-Online-Planning-/blob/a03ed1f43899ce9c5fd3154e3895dc781b1dbcd4/structure.png)

<!-- ## Download

You can [download](https://github.com/amitmerchant1990/electron-markdownify/releases/tag/v1.2.0) the latest installable version of Markdownify for Windows, macOS and Linux.

## Emailware

Markdownify is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <bullredeyes@gmail.com> about anything you'd want to say about this software. I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/)

## Related

[markdownify-web](https://github.com/amitmerchant1990/markdownify-web) - Web version of Markdownify

## Support

<a href="https://www.buymeacoffee.com/5Zn8Xh3l9" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<p>Or</p> 

<a href="https://www.patreon.com/amitmerchant">
	<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a> -->

<!-- ## You may also like...

- [Pomolectron](https://github.com/amitmerchant1990/pomolectron) - A pomodoro app
- [Correo](https://github.com/amitmerchant1990/correo) - A menubar/taskbar Gmail App for Windows and macOS

## License

MIT

---

> [amitmerchant.com](https://www.amitmerchant.com) &nbsp;&middot;&nbsp;
> GitHub [@amitmerchant1990](https://github.com/amitmerchant1990) &nbsp;&middot;&nbsp;
> Twitter [@amit_merchant](https://twitter.com/amit_merchant) -->

