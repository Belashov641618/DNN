class Formater:
    def __init__(self):
        self.EngineeringPrefixes = ['a', 'f', 'n', 'mk', 'm', '', 'K', 'M', 'G', 'T']
        self.EngineeringCenter = 5
        self.ScientificNumbers = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','⁻']

        DefaultStile = {
            'font': 'Times New Roman',
            'fontsize': 12,
            'fontweight': 'normal',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        HeaderStyle = {
            'font': 'Times New Roman',
            'fontsize': 16,
            'fontweight': 'bold',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        BigHeaderStyle = {
            'font': 'Times New Roman',
            'fontsize': 20,
            'fontweight': 'bold',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        CaptionStyle = {
            'font': 'Times New Roman',
            'fontsize': 8,
            'fontweight': 'normal',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        SmallCaptionStyle = {
            'font': 'Times New Roman',
            'fontsize': 6,
            'fontweight': 'light',
            'rotation': 0,
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
            'c': 'black'}
        self.Styles = {
            'Default': DefaultStile,
            'Header': HeaderStyle,
            'BigHeader': BigHeaderStyle,
            'Caption': CaptionStyle,
            'SmallCaption': SmallCaptionStyle
        }

    def Engineering(self, value, unit='', precision=3):
        c = self.EngineeringCenter
        sign = 1
        if value < 0:
            sign = -1
            value = -value
        while value <= 1.0 and c != 0:
            value *= 1000
            c -= 1
        while value >= 1000.0 and c != len(self.EngineeringPrefixes) - 1:
            value /= 1000
            c += 1
        return str(round(sign*value, precision)) + ' ' + self.EngineeringPrefixes[c] + unit
    def Engineering_Separated(self, value, unit=''):
        c = self.EngineeringCenter
        sign = 1
        if value == 0:
            return self.EngineeringPrefixes[c] + unit, 1000 ** (self.EngineeringCenter - c)
        if value < 0:
            sign = -1
            value = -value
        while value <= 1.0 and c != 0:
            value *= 1000
            c -= 1
        while value >= 1000.0 and c != len(self.EngineeringPrefixes) - 1:
            value /= 1000
            c += 1
        return self.EngineeringPrefixes[c] + unit, 1000**(self.EngineeringCenter-c)

    def Scientific(self, value, unit='', precision=3):
        if value == 0:
            return '0 ' + unit
        c = 0
        sign = 1
        if (value < 0):
            sign = -1
            value = -value
        while value <= 1.0:
            value *= 10
            c -= 1
        while value >= 10.0:
            value /= 10
            c += 1
        ps = ''
        if c < 0:
            ps += self.ScientificNumbers[10]
            c = -c
        ps_ = ''
        if c == 0:
            return str(round(sign*value, precision)) + ' ' + unit
        while c != 0:
            ps_ += self.ScientificNumbers[int(c%10)]
            c = int(c/10)
        ps += ps_[::-1]
        return str(round(sign*value, precision)) + '·10' + ps + ' ' + unit
    def Scientific_Separated(self, value, unit=''):
        if value == 0:
            return '0 ' + unit
        c = 0
        sign = 1
        if (value < 0):
            sign = -1
            value = -value
        while value <= 1.0:
            value *= 10
            c -= 1
        while value >= 10.0:
            value /= 10
            c += 1
        power = c
        ps = ''
        if c < 0:
            ps += self.ScientificNumbers[10]
            c = -c
        ps_ = ''
        if c==0:
            return '1' + ' ' + unit, 1.0
        while c != 0:
            ps_ += self.ScientificNumbers[int(c%10)]
            c = int(c/10)
        ps += ps_[::-1]
        return '10' + ps + ' ' + unit, 10**(-power)

    def Time(self, seconds=0, days=0, hours=0, minutes=0, millis=0, micros=0, nanos=0):
        time = seconds + millis*1.0E-3 + micros*1.0E-6 + nanos*1.0E-9
        days = int(time/86400)
        time -= days*86400
        hours = int(time/3600)
        time -= hours*3600
        minutes = int(time/60)
        time-= minutes*60
        seconds = int(time/1)
        time -= seconds*1
        millis = int(time*1000)
        time -= millis/1000
        micros = int(time*1000000)
        time -= micros/1000000
        nanos = int(time*1000000000)
        time -= nanos*1000000000
        time_array = [int(days),int(hours),int(minutes),int(seconds),int(millis),int(micros),int(nanos)]
        time_units = ['d','h','m','s','ms','us','ns']
        string = ''
        for i in range(len(time_array)):
            if time_array[i] != 0:
                string = str(time_array[i]) + time_units[i]
                if i != (len(time_array)-1) and time_array[i+1] != 0:
                    return string + str(time_array[i+1]) + time_units[i+1]
                return string
        return '0us0ns'

    def TextStyles(self):
        return self.Styles.copy()
    def Text(self, style, parameters={}):
        CurrentStyle = self.Styles[style].copy()
        CurrentStyle.update(parameters.items())
        return CurrentStyle

    def WrappedText(self, N, text):
        text_ = ''
        for i, l in enumerate(text):
            if (i+1) % N == 0:
                if l == ' ':
                    text_ += '\n'
                else:
                    for i in reversed(range(len(text_))):
                        if text_[i] == ' ':
                            text_ = text_[:i] + '\n' + text_[i:]
                            break
                    text_ += l
            else:
                text_ += l
        return text_

Format = Formater()