package test; /**
 * Created by Chih-Chen Kao on 1/9/2017.
 */

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.lwjgl.BufferUtils.createByteBuffer;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL11.*;
import static org.lwjgl.opencl.KHRICD.CL_PLATFORM_ICD_SUFFIX_KHR;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.system.Pointer.POINTER_SIZE;

// helper
import static org.lwjgl.system.windows.WinBase.TRUE;
import static test.InfoUtil.*;

public class OpenCLTest {





    private void initOpenCL(MemoryStack stack) {


        IntBuffer pi = stack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, pi));
        if ( pi.get(0) == 0 )
            throw new RuntimeException("No OpenCL platforms found.");

        PointerBuffer platforms = stack.mallocPointer(pi.get(0));
        checkCLError(clGetPlatformIDs(platforms, (IntBuffer)null));

        PointerBuffer ctxProps = stack.mallocPointer(3);
        ctxProps
                .put(0, CL_CONTEXT_PLATFORM)
                .put(2, 0);

        IntBuffer errcode_ret = stack.callocInt(1);
        for ( int p = 0; p < platforms.capacity(); p++ ) {

            long platform = platforms.get(p);
            ctxProps.put(1, platform);

            System.out.println("\n-------------------------");
            System.out.printf("NEW PLATFORM: [0x%X]\n", platform);

            CLCapabilities platformCaps = CL.createPlatformCapabilities(platform);

            printPlatformInfo(platform, "CL_PLATFORM_PROFILE", CL_PLATFORM_PROFILE);
            printPlatformInfo(platform, "CL_PLATFORM_VERSION", CL_PLATFORM_VERSION);
            printPlatformInfo(platform, "CL_PLATFORM_NAME", CL_PLATFORM_NAME);
            printPlatformInfo(platform, "CL_PLATFORM_VENDOR", CL_PLATFORM_VENDOR);
            printPlatformInfo(platform, "CL_PLATFORM_EXTENSIONS", CL_PLATFORM_EXTENSIONS);
            if ( platformCaps.cl_khr_icd )
                printPlatformInfo(platform, "CL_PLATFORM_ICD_SUFFIX_KHR", CL_PLATFORM_ICD_SUFFIX_KHR);
            System.out.println("");

            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, null, pi));

            PointerBuffer devices = stack.mallocPointer(pi.get(0));
            checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer)null));

            for ( int d = 0; d < devices.capacity(); d++ ) {

                long device = devices.get(d);
                CLCapabilities caps = CL.createDeviceCapabilities(device, platformCaps);

                System.out.printf("\n\t** NEW DEVICE: [0x%X]\n", device);

                System.out.println("\tCL_DEVICE_TYPE = " + getDeviceInfoLong(device, CL_DEVICE_TYPE));
                System.out.println("\tCL_DEVICE_VENDOR_ID = " + getDeviceInfoInt(device, CL_DEVICE_VENDOR_ID));
                System.out.println("\tCL_DEVICE_MAX_COMPUTE_UNITS = " + getDeviceInfoInt(device, CL_DEVICE_MAX_COMPUTE_UNITS));
                System.out
                        .println("\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = " + getDeviceInfoInt(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
                System.out.println("\tCL_DEVICE_MAX_WORK_GROUP_SIZE = " + getDeviceInfoPointer(device, CL_DEVICE_MAX_WORK_GROUP_SIZE));
                System.out.println("\tCL_DEVICE_MAX_CLOCK_FREQUENCY = " + getDeviceInfoInt(device, CL_DEVICE_MAX_CLOCK_FREQUENCY));
                System.out.println("\tCL_DEVICE_ADDRESS_BITS = " + getDeviceInfoInt(device, CL_DEVICE_ADDRESS_BITS));
                System.out.println("\tCL_DEVICE_AVAILABLE = " + (getDeviceInfoInt(device, CL_DEVICE_AVAILABLE) != 0));
                System.out.println("\tCL_DEVICE_COMPILER_AVAILABLE = " + (getDeviceInfoInt(device, CL_DEVICE_COMPILER_AVAILABLE) != 0));

                printDeviceInfo(device, "CL_DEVICE_NAME", CL_DEVICE_NAME);
                printDeviceInfo(device, "CL_DEVICE_VENDOR", CL_DEVICE_VENDOR);
                printDeviceInfo(device, "CL_DRIVER_VERSION", CL_DRIVER_VERSION);
                printDeviceInfo(device, "CL_DEVICE_PROFILE", CL_DEVICE_PROFILE);
                printDeviceInfo(device, "CL_DEVICE_VERSION", CL_DEVICE_VERSION);
                printDeviceInfo(device, "CL_DEVICE_EXTENSIONS", CL_DEVICE_EXTENSIONS);
                if ( caps.OpenCL11 )
                    printDeviceInfo(device, "CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION);

                CLContextCallback contextCB;
                long clContext = clCreateContext(ctxProps, device, contextCB = CLContextCallback.create((errinfo, private_info, cb, user_data) -> {
                    System.err.println("[LWJGL] cl_context_callback");
                    System.err.println("\tInfo: " + memUTF8(errinfo));
                }), NULL, errcode_ret);
                checkCLError(errcode_ret);

                // buffer

                long buffer = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 128, errcode_ret);
                checkCLError(errcode_ret);

                CLMemObjectDestructorCallback bufferCB1 = null;
                CLMemObjectDestructorCallback bufferCB2 = null;

                long subbuffer = NULL;
                CLMemObjectDestructorCallback subbufferCB = null;

                int errcode;

                CountDownLatch destructorLatch;

                if ( caps.OpenCL11 ) {
                    destructorLatch = new CountDownLatch(3);

                    errcode = clSetMemObjectDestructorCallback(buffer, bufferCB1 = CLMemObjectDestructorCallback.create((memobj, user_data) -> {
                        System.out.println("\t\tBuffer destructed (1): " + memobj);
                        destructorLatch.countDown();
                    }), NULL);
                    checkCLError(errcode);

                    errcode = clSetMemObjectDestructorCallback(buffer, bufferCB2 = CLMemObjectDestructorCallback.create((memobj, user_data) -> {
                        System.out.println("\t\tBuffer destructed (2): " + memobj);
                        destructorLatch.countDown();
                    }), NULL);
                    checkCLError(errcode);

                    try ( CLBufferRegion buffer_region = CLBufferRegion.malloc() ) {
                        buffer_region.origin(0);
                        buffer_region.size(64);

                        subbuffer = nclCreateSubBuffer(buffer,
                                CL_MEM_READ_ONLY,
                                CL_BUFFER_CREATE_TYPE_REGION,
                                buffer_region.address(),
                                memAddress(errcode_ret));
                        checkCLError(errcode_ret);
                    }

                    errcode = clSetMemObjectDestructorCallback(subbuffer, subbufferCB = CLMemObjectDestructorCallback.create((memobj, user_data) -> {
                        System.out.println("\t\tSub Buffer destructed: " + memobj);
                        destructorLatch.countDown();
                    }), NULL);
                    checkCLError(errcode);
                } else
                    destructorLatch = null;

                long exec_caps = getDeviceInfoLong(device, CL_DEVICE_EXECUTION_CAPABILITIES);

                if ( (exec_caps & CL_EXEC_NATIVE_KERNEL) == CL_EXEC_NATIVE_KERNEL ) {

                    System.out.println("\t\t-TRYING TO EXEC NATIVE KERNEL-");
                    long queue = clCreateCommandQueue(clContext, device, NULL, errcode_ret);

                    PointerBuffer ev = BufferUtils.createPointerBuffer(1);

                    ByteBuffer kernelArgs = createByteBuffer(4);
                    kernelArgs.putInt(0, 1337);

                    CLNativeKernel kernel;
                    errcode = clEnqueueNativeKernel(queue, kernel = CLNativeKernel.create(
                            args -> System.out.println("\t\tKERNEL EXEC argument: " + memByteBuffer(args, 4).getInt(0) + ", should be 1337")
                    ), kernelArgs, null, null, null, ev);
                    checkCLError(errcode);

                    long e = ev.get(0);

                    CountDownLatch latch = new CountDownLatch(1);

                    CLEventCallback eventCB;
                    errcode = clSetEventCallback(e, CL_COMPLETE, eventCB = CLEventCallback.create((event, event_command_exec_status, user_data) -> {
                        System.out.println("\t\tEvent callback status: " + getEventStatusName(event_command_exec_status));
                        latch.countDown();
                    }), NULL);
                    checkCLError(errcode);

                    try {
                        boolean expired = !latch.await(500, TimeUnit.MILLISECONDS);
                        if ( expired )
                            System.out.println("\t\tKERNEL EXEC FAILED!");
                    } catch (InterruptedException exc) {
                        exc.printStackTrace();
                    }

                    eventCB.free();

                    errcode = clReleaseEvent(e);
                    checkCLError(errcode);
                    kernel.free();

                    kernelArgs = createByteBuffer(POINTER_SIZE * 2);

                    kernel = CLNativeKernel.create(args -> {
                    });

                    long time = System.nanoTime();
                    int REPEAT = 1000;
                    for ( int i = 0; i < REPEAT; i++ ) {
                        clEnqueueNativeKernel(queue, kernel, kernelArgs, null, null, null, null);
                    }
                    clFinish(queue);
                    time = System.nanoTime() - time;

                    System.out.printf("\n\t\tEMPTY NATIVE KERNEL AVG EXEC TIME: %.4fus\n", (double)time / (REPEAT * 1000));

                    errcode = clReleaseCommandQueue(queue);
                    checkCLError(errcode);
                    kernel.free();
                }


                // Try To build program and kernel

                // ----------------- start

                ByteBuffer source;
                PointerBuffer arrayOfKernelSources;
                PointerBuffer lengths;

                try {

                    // check the kernel name
                    // check the buffer size
                    source = ioResourceToByteBuffer("test/test.cl", 4096);

                    arrayOfKernelSources = BufferUtils.createPointerBuffer(1);
                    lengths = BufferUtils.createPointerBuffer(1);

                } catch (IOException e) {
                    throw new RuntimeException(e);
                }


                arrayOfKernelSources.put(0, source);
                lengths.put(0, source.remaining());

                long clProgram = clCreateProgramWithSource(clContext, arrayOfKernelSources, lengths, errcode_ret);
                checkCLError(errcode_ret);

                CountDownLatch latch = new CountDownLatch(1);

                // disable 64bit floating point math if not available
                StringBuilder options = new StringBuilder("-cl-std=CL2.0");

                System.out.println("OpenCL COMPILER OPTIONS: " + options);

                CLProgramCallback buildCallback;

                errcode = clBuildProgram(clProgram, device, options, buildCallback = CLProgramCallback.create((program, user_data) -> {
                    System.out.println(String.format(
                            "The cl_program [0x%X] was built %s",
                            program,
                            getProgramBuildInfoInt(program, device, CL_PROGRAM_BUILD_STATUS) == CL_SUCCESS ? "successfully" : "unsuccessfully"
                    ));

                    String log = getProgramBuildInfoStringASCII(program, device, CL_PROGRAM_BUILD_LOG);
                    if ( !log.isEmpty() ) {
                        System.out.println(String.format("BUILD LOG:\n----\n%s\n-----", log));
                    }

                    latch.countDown();
                }), NULL);

                checkCLError(errcode);



                // Make sure the program has been built before proceeding
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }



                buildCallback.free();

                // init kernel with constants
                long clKernel = clCreateKernel(clProgram, "test", errcode_ret);
                checkCLError(errcode_ret);


                // memory buffers



                // there are three ways to allocate memory in LWJGL:
                // 1. stack-based approach: org.lwjgl.system.MemoryStack
                // 2. Malloc: org.lwjgl.system.MemoryUtil
                // 3. ByteBuffer: org.lwjgl.BufferUtil
                // Efficiency ordered from top to bottom.

                final int bufferSize = 1024 * 1024 * 10;

                // the buffer is too large to be allocated on the stack
                //IntBuffer hostBufferA = stack.mallocInt(bufferSize);
                IntBuffer  hostBufferA = memAllocInt(bufferSize);

                for (int j = 0; j < hostBufferA.capacity(); ++j) {
                    hostBufferA.put(j, j);
                }

                // use host_ptr
                long bufferA = clCreateBuffer(clContext, CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR, hostBufferA, errcode_ret);
                checkCLError(errcode_ret);


                //
                //IntBuffer hostBufferB = stack.mallocInt(bufferSize);
                IntBuffer  hostBufferB = memAllocInt(bufferSize);

                for (int j = 0; j < hostBufferB.capacity(); ++j) {
                    hostBufferB.put(j, 10);
                }

                // copy host_ptr
                long bufferB = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, hostBufferB, errcode_ret);
                checkCLError(errcode_ret);

                //
                //IntBuffer hostBufferC = stack.mallocInt(bufferSize);
                IntBuffer  hostBufferC = memAllocInt(bufferSize);

                long bufferC = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , hostBufferC, errcode_ret);
                checkCLError(errcode_ret);



                // CLMemObjectDestructorCallback ?

                // set kernel args
                checkCLError(clSetKernelArg1p(clKernel, 0, bufferA));
                checkCLError(clSetKernelArg1p(clKernel, 1, bufferB));
                checkCLError(clSetKernelArg1p(clKernel, 2, bufferC));


                // create CQ
                long queueCL = clCreateCommandQueue(clContext, device, NULL, errcode_ret);
                checkCLError(errcode_ret);




                PointerBuffer kernel2DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
                kernel2DGlobalWorkSize.put(0, bufferSize);

                long execTime = System.nanoTime();

                errcode = clEnqueueNDRangeKernel(queueCL, clKernel, 1, null,
                        kernel2DGlobalWorkSize, null, null, null);
                checkCLError(errcode);

                execTime = System.nanoTime() - execTime;

                System.out.printf("KERNEL EXEC TIME: %.4fus\n", (double)execTime);


                // read buffer

                // no need to write or read ??
                errcode = clEnqueueReadBuffer(queueCL, bufferC, TRUE, 0, hostBufferC, null, null);
                checkCLError(errcode);


                // memory map


                // beware of the buffer size
                ByteBuffer hostMappedByteBufferA = clEnqueueMapBuffer(queueCL, bufferA, TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bufferSize * Integer.BYTES, null, null, errcode_ret, null );
                IntBuffer hostMappedBufferA = hostMappedByteBufferA.asIntBuffer();
                checkCLError(errcode_ret);






                // verify the result

                System.out.println("Verify the result");
                for (int j = 0; j < hostMappedBufferA.capacity(); ++j) {
                    int valueA = hostMappedBufferA.get(j);
                    int valueC = hostBufferC.get(j);

                    //System.out.printf("[%d]:%d:%d, ", j, valueA, valueC);

                    if (valueA != valueC)
                        throw new RuntimeException(String.format("verification fail at %d", j));
                }

                // unmap


                errcode = clEnqueueUnmapMemObject(queueCL, bufferA, hostMappedByteBufferA, null, null);
                checkCLError(errcode);



                // free
                memFree(hostBufferA);
                memFree(hostBufferB);
                memFree(hostBufferC);

                System.out.println("PASSED !");



                // clean up

                errcode = clReleaseProgram(clProgram);
                checkCLError(errcode);

                errcode = clReleaseCommandQueue(queueCL);
                checkCLError(errcode);


                errcode = clReleaseKernel(clKernel);
                checkCLError(errcode);


                //  ------------------ rest of the world

                System.out.println();

                if ( subbuffer != NULL ) {
                    errcode = clReleaseMemObject(subbuffer);
                    checkCLError(errcode);
                }

                errcode = clReleaseMemObject(buffer);
                checkCLError(errcode);

                if ( destructorLatch != null ) {
                    // mem object destructor callbacks are called asynchronously on Nvidia

                    try {
                        destructorLatch.await();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    subbufferCB.free();

                    bufferCB2.free();
                    bufferCB1.free();
                }

                errcode = clReleaseContext(clContext);
                checkCLError(errcode);

                contextCB.free();
            }
        }





    }


    private static ByteBuffer resizeBuffer(ByteBuffer buffer, int newCapacity) {
        ByteBuffer newBuffer = BufferUtils.createByteBuffer(newCapacity);
        buffer.flip();
        newBuffer.put(buffer);
        return newBuffer;
    }

    public static ByteBuffer ioResourceToByteBuffer(String resource, int bufferSize) throws IOException {
        ByteBuffer buffer;

        Path path = Paths.get(resource);

        if ( Files.isReadable(path) ) {
            try (SeekableByteChannel fc = Files.newByteChannel(path)) {
                buffer = createByteBuffer((int)fc.size() + 1);
                while ( fc.read(buffer) != -1 ) ;
            }
        } else {
            try (
                    // test, need to check the location
                    InputStream source = test.OpenCLTest.class.getClassLoader().getResourceAsStream(resource);
                    ReadableByteChannel rbc = Channels.newChannel(source)
            ) {
                buffer = createByteBuffer(bufferSize);

                while ( true ) {
                    int bytes = rbc.read(buffer);
                    if ( bytes == -1 )
                        break;
                    if ( buffer.remaining() == 0 )
                        buffer = resizeBuffer(buffer, buffer.capacity() * 2);
                }
            }
        }

        buffer.flip();
        return buffer;
    }


    public void run() {


        try ( MemoryStack stack = stackPush() ) {
            initOpenCL(stack);
        }


    }





}
