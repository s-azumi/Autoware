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

WaypointHandlerMarker::WaypointHandlerMarker(boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server_ptr,int marker_id)
{
    server_ptr_ = server_ptr;
    marker_id_ = marker_id;
    visualization_msgs::InteractiveMarker int_marker;
    int_marker.header.frame_id = "world";
    int_marker.scale = 1;
    int_marker.name = "waypoint_editor_marker";
    int_marker.description = "waypoint editor marker";
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
    server_ptr_->insert(int_marker);
    server_ptr_->setCallback(int_marker.name, boost::bind(&WaypointHandlerMarker::processFeedback,this,_1));
    initMenu();
    server_ptr_->applyChanges();
    ROS_INFO_STREAM("add new marker!!");
}

WaypointHandlerMarker::~WaypointHandlerMarker()
{

}

void WaypointHandlerMarker::initMenu()
{   
    menu_handler_.setCheckState(menu_handler_.insert( "Add New Hadnler", boost::bind(&WaypointHandlerMarker::addHandlerCallback,this,_1)), 
        interactive_markers::MenuHandler::UNCHECKED);
    menu_handler_.apply(*server_ptr_, "waypoint_editor_marker");
    return;
}

void WaypointHandlerMarker::updateMarkerId(int marker_id)
{
    marker_id_ = marker_id;
    return;
}

void WaypointHandlerMarker::addHandlerCallback( const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback)
{
    return;
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