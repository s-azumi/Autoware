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

#include "waypoint_handler_marker.h"

WaypointHandlerMarker::WaypointHandlerMarker(interactive_markers::InteractiveMarkerServer& server_,int marker_id)
{
    visualization_msgs::InteractiveMarker int_marker;
    int_marker.header.frame_id = "world";
    int_marker.scale = 1;
    int_marker.name = "waypoint_editor_marker"+std::to_string(marker_id);
    int_marker.description = "waypoint editor marker"+std::to_string(marker_id);
    makeBoxControl(int_marker);
    visualization_msgs::InteractiveMarkerControl control;
    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 1;
    control.orientation.z = 0;
    control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_ROTATE;
    int_marker.controls.push_back(control);
    control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);
    server_.insert(int_marker);
    server_.setCallback(int_marker.name, boost::bind(&WaypointHandlerMarker::processFeedback,this,_1));
    server_.applyChanges();
    ROS_INFO_STREAM("add new marker!!");
}

WaypointHandlerMarker::~WaypointHandlerMarker()
{

}

visualization_msgs::InteractiveMarkerControl& WaypointHandlerMarker::makeBoxControl(visualization_msgs::InteractiveMarker &msg)
{
    visualization_msgs::InteractiveMarkerControl control;
    control.always_visible = true;
    control.markers.push_back(makeArrow(msg));
    msg.controls.push_back(control);
    return msg.controls.back();
}

visualization_msgs::Marker WaypointHandlerMarker::makeArrow(visualization_msgs::InteractiveMarker &msg)
{
    visualization_msgs::Marker marker;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.scale.x = msg.scale * 0.60;
    marker.scale.y = msg.scale * 0.30;
    marker.scale.z = msg.scale * 0.30;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    return marker;
}

void WaypointHandlerMarker::processFeedback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback)
{
    return;
}