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
#include "waypoint_handler_marker.h"

//headers in boost
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

class WaypointEditor
{
public:
    WaypointEditor(ros::NodeHandle nh,ros::NodeHandle pnh);
    ~WaypointEditor();
private:
    boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server_ptr_;
    void addMarker(int marker_id);
    std::vector<WaypointHandlerMarker> markers_;
};