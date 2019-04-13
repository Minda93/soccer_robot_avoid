
"use strict";

let simulation_strategy = require('./simulation_strategy.js');
let StrategyInfo = require('./StrategyInfo.js');
let CoachInfo = require('./CoachInfo.js');
let WorldModelInfo = require('./WorldModelInfo.js');
let RobotInfo = require('./RobotInfo.js');
let MotionCmd = require('./MotionCmd.js');
let BallInfo = require('./BallInfo.js');
let VelCmd = require('./VelCmd.js');
let PassCommands = require('./PassCommands.js');
let currentCmd = require('./currentCmd.js');
let Angle = require('./Angle.js');
let Point3d = require('./Point3d.js');
let ObstaclesInfo = require('./ObstaclesInfo.js');
let OdoInfo = require('./OdoInfo.js');
let OminiVisionInfo = require('./OminiVisionInfo.js');
let TargetInfo = require('./TargetInfo.js');
let PPoint = require('./PPoint.js');
let BallInfo3d = require('./BallInfo3d.js');
let MotorInfo = require('./MotorInfo.js');
let FrontBallInfo = require('./FrontBallInfo.js');
let Point2d = require('./Point2d.js');

module.exports = {
  simulation_strategy: simulation_strategy,
  StrategyInfo: StrategyInfo,
  CoachInfo: CoachInfo,
  WorldModelInfo: WorldModelInfo,
  RobotInfo: RobotInfo,
  MotionCmd: MotionCmd,
  BallInfo: BallInfo,
  VelCmd: VelCmd,
  PassCommands: PassCommands,
  currentCmd: currentCmd,
  Angle: Angle,
  Point3d: Point3d,
  ObstaclesInfo: ObstaclesInfo,
  OdoInfo: OdoInfo,
  OminiVisionInfo: OminiVisionInfo,
  TargetInfo: TargetInfo,
  PPoint: PPoint,
  BallInfo3d: BallInfo3d,
  MotorInfo: MotorInfo,
  FrontBallInfo: FrontBallInfo,
  Point2d: Point2d,
};
