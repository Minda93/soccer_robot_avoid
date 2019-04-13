// Generated by gencpp from file nubot_common/Point2d.msg
// DO NOT EDIT!


#ifndef NUBOT_COMMON_MESSAGE_POINT2D_H
#define NUBOT_COMMON_MESSAGE_POINT2D_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace nubot_common
{
template <class ContainerAllocator>
struct Point2d_
{
  typedef Point2d_<ContainerAllocator> Type;

  Point2d_()
    : x(0.0)
    , y(0.0)  {
    }
  Point2d_(const ContainerAllocator& _alloc)
    : x(0.0)
    , y(0.0)  {
  (void)_alloc;
    }



   typedef float _x_type;
  _x_type x;

   typedef float _y_type;
  _y_type y;





  typedef boost::shared_ptr< ::nubot_common::Point2d_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::nubot_common::Point2d_<ContainerAllocator> const> ConstPtr;

}; // struct Point2d_

typedef ::nubot_common::Point2d_<std::allocator<void> > Point2d;

typedef boost::shared_ptr< ::nubot_common::Point2d > Point2dPtr;
typedef boost::shared_ptr< ::nubot_common::Point2d const> Point2dConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::nubot_common::Point2d_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::nubot_common::Point2d_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace nubot_common

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'nubot_common': ['/home/minda/Minda/github/soccer_robot_avoid/src/ros_gazebo/nubot/nubot_common/msgs'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::nubot_common::Point2d_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::nubot_common::Point2d_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::nubot_common::Point2d_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::nubot_common::Point2d_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::nubot_common::Point2d_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::nubot_common::Point2d_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::nubot_common::Point2d_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ff8d7d66dd3e4b731ef14a45d38888b6";
  }

  static const char* value(const ::nubot_common::Point2d_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xff8d7d66dd3e4b73ULL;
  static const uint64_t static_value2 = 0x1ef14a45d38888b6ULL;
};

template<class ContainerAllocator>
struct DataType< ::nubot_common::Point2d_<ContainerAllocator> >
{
  static const char* value()
  {
    return "nubot_common/Point2d";
  }

  static const char* value(const ::nubot_common::Point2d_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::nubot_common::Point2d_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32 x\n\
float32 y\n\
";
  }

  static const char* value(const ::nubot_common::Point2d_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::nubot_common::Point2d_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.x);
      stream.next(m.y);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Point2d_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::nubot_common::Point2d_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::nubot_common::Point2d_<ContainerAllocator>& v)
  {
    s << indent << "x: ";
    Printer<float>::stream(s, indent + "  ", v.x);
    s << indent << "y: ";
    Printer<float>::stream(s, indent + "  ", v.y);
  }
};

} // namespace message_operations
} // namespace ros

#endif // NUBOT_COMMON_MESSAGE_POINT2D_H
