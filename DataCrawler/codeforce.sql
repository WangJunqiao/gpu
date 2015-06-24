/*
Navicat MySQL Data Transfer

Source Server         : localhost
Source Server Version : 50614
Source Host           : localhost:3306
Source Database       : codeforce

Target Server Type    : MYSQL
Target Server Version : 50614
File Encoding         : 65001

Date: 2015-06-24 16:38:55
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `status`
-- ----------------------------
DROP TABLE IF EXISTS `status`;
CREATE TABLE `status` (
  `submissionId` int(8) NOT NULL,
  `contestId` varchar(8) NOT NULL,
  `timestamp` varchar(32) NOT NULL,
  `coderId` varchar(64) NOT NULL,
  `language` varchar(32) NOT NULL,
  `result` varchar(64) NOT NULL,
  `runTime` varchar(16) NOT NULL,
  `runMemory` varchar(16) NOT NULL,
  PRIMARY KEY (`submissionId`),
  KEY `submissionIdIndex` (`submissionId`) USING HASH
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of status
-- ----------------------------

-- ----------------------------
-- Table structure for `visited_page`
-- ----------------------------
DROP TABLE IF EXISTS `visited_page`;
CREATE TABLE `visited_page` (
  `pageId` int(11) NOT NULL,
  PRIMARY KEY (`pageId`),
  UNIQUE KEY `index` (`pageId`) USING HASH
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of visited_page
-- ----------------------------

-- ----------------------------
-- Table structure for `visited_submissionid`
-- ----------------------------
DROP TABLE IF EXISTS `visited_submissionid`;
CREATE TABLE `visited_submissionid` (
  `submissionId` int(11) NOT NULL,
  PRIMARY KEY (`submissionId`),
  UNIQUE KEY `index` (`submissionId`) USING HASH
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of visited_submissionid
-- ----------------------------
