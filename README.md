# DdddOCR in C#

NOTE:为了保证运行速度, CORE建议全局初始化一次

```c#

namespace DdddOCR.Test;

internal class Program
{
    private static OCRCore Core;
    static void Main(string[] args)
    {
        Core = new();
        var text = Core.DetectText(@".\TestPicture\code1.png");
        Console.WriteLine(text);
        text = Core.DetectText(@".\TestPicture\bigPic.JPG");
        Console.WriteLine(text);
        text = Core.DetectText(@".\TestPicture\TestbigAndPrintFont.jpg");
        Console.WriteLine(text);
        text = Core.DetectText(@".\TestPicture\CHs.bmp");
        Console.WriteLine(text);
    }
}
```

