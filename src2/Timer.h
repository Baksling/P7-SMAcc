#ifndef TIMER_H
#define TIMER_H

class timer
{
private:
    double time_;
public:
    timer(double time = 0);
    double get_time();
    
};

#endif // TIMER_H
