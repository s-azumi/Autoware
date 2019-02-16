#ifndef WAYPOINT_HANDLER_MARKER_H_INCLUDED
#define WAYPOINT_HANDLER_MARKER_H_INCLUDED

/*
 * Copyright 2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * v1.0 Masaya Kataoka
 */

//headers in ROS
#include <ros/ros.h>
#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/menu_handler.h>

//headers in stl
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

class WaypointHandlerMarker
{
public:
    WaypointHandlerMarker(interactive_markers::InteractiveMarkerServer& server_,int marker_id);
    ~WaypointHandlerMarker();
private:
    void processFeedback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback);
    visualization_msgs::Marker makeArrow(visualization_msgs::InteractiveMarker &msg);
    visualization_msgs::InteractiveMarkerControl& makeBoxControl(visualization_msgs::InteractiveMarker &msg);
};

#endif  //WAYPOINT_HANDLER_MARKER_H_INCLUDED