using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DdddOCR;

/// <summary>
/// DDDDOCR 的纯 C# 实现
/// </summary>
public class OCRCore : IDisposable
{
    private readonly InferenceSession _inferenceSession;
    private readonly SessionOptions _sessionOptions;
    private readonly List<char> _charset;
    private bool disposedValue;

    /// <summary>
    /// 初始化实例
    /// </summary>
    public OCRCore()
    {
        _sessionOptions = new();
        _sessionOptions.AppendExecutionProvider_CPU();
        _inferenceSession = new(Properties.Resources.Common_Old, _sessionOptions);
        _charset = new List<char>
        {
            char.MinValue
        };
        _charset.AddRange(Properties.Resources.CharSet.ToCharArray());
    }

    /// <summary>
    /// 提取文字
    /// </summary>
    /// <param name="imageBytes"> 图片字节组 </param>
    /// <returns> 文本 </returns>
    public string DetectText(byte[] imageBytes)
    {
        return Pipeline(Image.Load(imageBytes));
    }

    /// <summary>
    /// 提取文字
    /// </summary>
    /// <param name="imageBytes"> 图片字节组 </param>
    /// <returns> 文本 </returns>
    public string DetectText(ReadOnlySpan<byte> imageBytes)
    {
        return Pipeline(Image.Load(imageBytes));
    }

    /// <summary>
    /// 提取文字
    /// </summary>
    /// <param name="imageFilePath"> 图片位置 </param>
    /// <returns> 文本 </returns>
    public string DetectText(string imageFilePath)
    {
        return Pipeline(Image.Load(imageFilePath));
    }

    /// <summary>
    /// 提取文字
    /// </summary>
    /// <param name="imageStream"> 图像流 </param>
    /// <returns> 文本 </returns>
    public string DetectText(Stream imageStream)
    {
        return Pipeline(Image.Load(imageStream));
    }

    private string Pipeline(Image originImage)
    {
        using var ms = new MemoryStream();
        originImage.SaveAsPng(ms);
        return SessionRun(PrepareProcessing(Image.Load<Argb32>(ms.ToArray())));
    }

    private static List<NamedOnnxValue> PrepareProcessing(Image<Argb32> originImage)
    {
        if (originImage.Width * originImage.Height == 0)
        {
            throw new Exception($"The input image size is 0 or empty ' {nameof(originImage)} '");
        }

        originImage.Mutate(img_ => img_
            .Grayscale()
            .Resize(originImage.Width / (originImage.Height / 64), 64, false)
            );
        var input = new DenseTensor<float>(new[] { 1, 1, 64, originImage.Width });

        originImage.ProcessPixelRows(x =>
        {
            for (int height = 0; height <= x.Height - 1; height++)
            {
                var row = x.GetRowSpan(height);
                for (int width = 0; width <= row.Length - 1; width++)
                {
                    input[0, 0, height, width] = ((row[width].B / 255F) - 0.5F) * 2F;
                }
            }
        });
        return new() { NamedOnnxValue.CreateFromTensor("input1", input) };
    }

    private string SessionRun(List<NamedOnnxValue> inputImageOnnxValue)
    {
        using var sessionResult = _inferenceSession.Run(inputImageOnnxValue);
        var tensorArray = sessionResult.ToArray().First().Value as DenseTensor<long>;
        var result = new List<char>();
        tensorArray.ToList().ForEach(x => result.Add(_charset[(int)x]));
        return string.Join(char.MinValue, result);
    }

    /// <summary>
    /// 回收方法
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                _inferenceSession.Dispose();
                _sessionOptions.Dispose();
            }

            disposedValue = true;
        }
    }
    /// <summary>
    /// 自动销毁实例
    /// </summary>
    ~OCRCore()
    {
        Dispose(disposing: false);
    }
    /// <summary>
    /// 手动销毁实例
    /// </summary>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}