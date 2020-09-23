import numpy as np 

def one_year():
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum_sum = np.cumsum(month_lengths)


    day_inds = range(1,366)

    week_days = range(1, 8)

    total_ws = 0
    end_ws = 0
    month_inds = []
    for i in day_inds:
        # check if wednesday
        day = i % 7
        if day == 3:
            total_ws += 1
            # check if last day of month
            for j in range(12):
                month = cum_sum[j]
                if i == month:
                    month_inds.append(j)
                    end_ws += 1

    print(f"Total Wednesdays: {total_ws}")
    print(f"End Wednesdays: {end_ws}")
    print(f"Percent end Wednesdays: {end_ws/total_ws * 100}")

    print(f"Months: {month_inds}")

def four_years():
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap_year = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    four_year = np.concatenate([month_lengths, month_lengths, month_lengths, leap_year])
    cum_sum = np.cumsum(four_year)

    n_days = 365 * 3 + 366
    day_inds = range(1, n_days + 1)

    total_ws = 0
    end_ws = 0
    month_inds = []
    for i in day_inds:
        # check if wednesday
        day = i % 7
        if day == 3:
            total_ws += 1
            # check if last day of month
            for j in range(48):
                month = cum_sum[j]
                
                if i == month:
                    month_inds.append(j)
                    end_ws += 1
    print(f"Last day: {day_inds[-1]%7}")



    print(f"Four years")
    print(f"Total Wednesdays: {total_ws}")
    print(f"End Wednesdays: {end_ws}")
    print(f"Percent end Wednesdays: {end_ws/total_ws * 100}")

    print(f"Months: {month_inds}")

def all_years():
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap_year = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    four_year = np.concatenate([month_lengths, month_lengths, month_lengths, leap_year])
    cum_sum = np.cumsum(four_year)

    n_days = 365 * 3 + 366

    total_ws = 0
    end_ws = 0
    month_inds = []

    first_day = 1
    for a in range(24):
        day_inds = range(first_day, n_days + first_day)
        for i in day_inds:
            # check if wednesday
            day = i % 7
            if day == 3:
                total_ws += 1
                # check if last day of month
                for j in range(48):
                    month = cum_sum[j]
                    
                    if i == month:
                        month_inds.append(j)
                        end_ws += 1
        first_day = (day_inds[-1] + 1)%7

    day_inds = range(first_day, 365*4 + first_day)
    for i in day_inds:
        # check if wednesday
        day = i % 7
        if day == 3:
            total_ws += 1
            # check if last day of month
            for j in range(48):
                month = cum_sum[j]
                
                if i == month:
                    month_inds.append(j)
                    end_ws += 1
    last_day = (day_inds[-1])%7
    

    print(f"Last day: {last_day}")



    print(f"Four years")
    print(f"Total Wednesdays: {total_ws}")
    print(f"End Wednesdays: {end_ws}")
    print(f"Percent end Wednesdays: {end_ws/total_ws * 100}")

    print(f"Months: {month_inds}")


one_year()
# four_years()
all_years()
